#!/usr/bin/env python3
"""
SAM3 BEV 投影节点 (bev_projector_node)

将深度图与 SAM3 分割结果联合投影到俯视 (Bird's Eye View) 坐标系，
并通过三种话题在 RViz2 中可视化：

订阅：
  • /camera/camera/aligned_depth_to_color/image_raw    — 对齐深度图 (16UC1, mm)
  • /camera/camera/color/image_raw                     — 彩色图 (bgr8)
  • /camera/camera/aligned_depth_to_color/camera_info  — 相机内参（独立订阅）
  • /sam3/detections_3d                                — 含3D位置的检测结果

发布：
  • /sam3/bev/pointcloud  — sensor_msgs/PointCloud2
      按分割类着色的三维点云（相机坐标系），可在 RViz2 俯视查看
  • /sam3/bev/markers     — visualization_msgs/MarkerArray
      每个检测目标的 3D 标注（圆球 + 文字标签）
  • /sam3/bev/image       — sensor_msgs/Image
      BEV 光栅化俯视图（XZ 平面，X 左右、Z 向深）

坐标约定（相机光学帧）：
  X → 右，Y → 下，Z → 前（深度方向）
  BEV 投影到 XZ 平面（忽略 Y / 高度分量）
"""

import copy
import struct

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from sam3_msgs.msg import DetectionArray


# ── 固定类颜色调板（RGB，每类对应一个鲜明颜色） ────────────────────────────────
_CLASS_PALETTE_RGB: list[tuple[int, int, int]] = [
    (220,  20,  60),  # 红 Crimson
    ( 50, 205,  50),  # 绿 LimeGreen
    (  0, 120, 255),  # 蓝 DodgerBlue
    (255, 165,   0),  # 橙 Orange
    (148,   0, 211),  # 紫 Violet
    (  0, 206, 209),  # 青 DarkTurquoise
    (255,  20, 147),  # 粉 DeepPink
    (255, 215,   0),  # 金 Gold
    (  0, 255, 127),  # 薄荷 SpringGreen
    (255,  99,  71),  # 番茄 Tomato
    (100, 149, 237),  # 矢车菊蓝 CornflowerBlue
    (154, 205,  50),  # 黄绿 YellowGreen
]


def _class_color_rgb(class_id: int) -> tuple[int, int, int]:
    return _CLASS_PALETTE_RGB[class_id % len(_CLASS_PALETTE_RGB)]


def _class_color_bgr(class_id: int) -> tuple[int, int, int]:
    r, g, b = _class_color_rgb(class_id)
    return (b, g, r)


def _class_color_rgba(class_id: int, alpha: float = 0.8) -> ColorRGBA:
    r, g, b = _class_color_rgb(class_id)
    return ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=alpha)


def _pack_rgb_float(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """将 uint8 R/G/B 数组打包为 float32（RViz2 RGB 点云格式）。"""
    packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    return packed.view(np.float32)


def _reconstruct_mask(
    contour_points,           # list of Point2D (x, y in mask space)
    mask_h: int,
    mask_w: int,
) -> np.ndarray:
    """从轮廓点重建二值掩膜图，返回 (mask_h, mask_w) uint8 [0/255]。"""
    canvas = np.zeros((mask_h, mask_w), dtype=np.uint8)
    if not contour_points:
        return canvas
    pts = np.array([[int(p.x), int(p.y)] for p in contour_points], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    return canvas


class BEVProjectorNode(Node):
    """将 SAM3 检测结果与深度图联合投影到 BEV 坐标系并发布可视化话题。"""

    def __init__(self):
        super().__init__("bev_projector_node")

        # ── 参数声明 ──────────────────────────────────────────────────────────
        _str  = ParameterType.PARAMETER_STRING
        _int  = ParameterType.PARAMETER_INTEGER
        _dbl  = ParameterType.PARAMETER_DOUBLE
        _bool = ParameterType.PARAMETER_BOOL

        def _p(name, default, ptype, desc):
            self.declare_parameter(name, default, ParameterDescriptor(type=ptype, description=desc))

        _p("depth_image_topic",   "/camera/camera/aligned_depth_to_color/image_raw",  _str,  "对齐深度图话题")
        _p("color_image_topic",   "/camera/camera/color/image_raw",                    _str,  "彩色图话题")
        _p("camera_info_topic",   "/camera/camera/aligned_depth_to_color/camera_info", _str,  "相机内参话题")
        _p("detections_3d_topic", "/sam3/detections_3d",                               _str,  "含3D坐标的检测结果话题")
        _p("frame_id",            "camera_color_optical_frame",                         _str,  "输出坐标系 frame_id")
        _p("queue_size",          20,    _int,  "话题缓冲队列大小")
        _p("sync_slop",           0.5,   _dbl,  "时间同步容差（秒）")
        _p("depth_scale",         0.001, _dbl,  "深度换算系数（mm → m）")
        _p("max_depth",           10.0,  _dbl,  "有效深度上限（米）")
        _p("imgsz",               518,   _int,  "SAM3 推理分辨率（需与 sam3_node 一致）")
        _p("downsample_step",     2,     _int,  "点云下采样步长（1=不跳像素，2=每隔1像素采一点）")
        _p("bev_x_range",         5.0,   _dbl,  "BEV 图像 X 轴单侧范围（米，图像宽 = 2×range）")
        _p("bev_z_max",           10.0,  _dbl,  "BEV 图像 Z 轴最大深度（米）")
        _p("bev_resolution",      0.02,  _dbl,  "BEV 图像每像素代表的物理距离（米/像素）")
        _p("marker_lifetime_sec", 0.5,   _dbl,  "RViz2 Marker 存活时间（秒），0 = 永久")
        _p("colorize_with_class", True,  _bool, "True：按类着色点云；False：保留原始 RGB 颜色")

        # ── 读取参数 ──────────────────────────────────────────────────────────
        gp = lambda n: self.get_parameter(n).value
        depth_topic     = gp("depth_image_topic")
        color_topic     = gp("color_image_topic")
        info_topic      = gp("camera_info_topic")
        det3d_topic     = gp("detections_3d_topic")
        self.frame_id   = gp("frame_id")
        queue_size      = int(gp("queue_size"))
        sync_slop       = float(gp("sync_slop"))
        self.depth_scale = float(gp("depth_scale"))
        self.max_depth  = float(gp("max_depth"))
        self.imgsz      = int(gp("imgsz"))
        self.step       = max(1, int(gp("downsample_step")))
        self.bev_x_range   = float(gp("bev_x_range"))
        self.bev_z_max     = float(gp("bev_z_max"))
        self.bev_res       = float(gp("bev_resolution"))
        marker_lifetime    = float(gp("marker_lifetime_sec"))
        self.colorize_class = bool(gp("colorize_with_class"))

        # Marker Duration
        secs = int(marker_lifetime)
        nsecs = int((marker_lifetime - secs) * 1e9)
        self._marker_lifetime = Duration(sec=secs, nanosec=nsecs)

        # BEV 图像尺寸（像素）
        self.bev_cols = int(round(2.0 * self.bev_x_range / self.bev_res))
        self.bev_rows = int(round(self.bev_z_max / self.bev_res))

        for k, v in [
            ("深度图话题", depth_topic), ("彩色图话题", color_topic),
            ("内参话题", info_topic), ("检测结果话题", det3d_topic),
            ("坐标系", self.frame_id), ("时间同步容差", sync_slop),
            ("BEV范围 X", f"±{self.bev_x_range}m"), ("BEV深度 Z", f"0~{self.bev_z_max}m"),
            ("BEV分辨率", f"{self.bev_res}m/px → {self.bev_cols}×{self.bev_rows}px"),
        ]:
            self.get_logger().info(f"  {k}: {v}")

        # ── 内参缓存 ──────────────────────────────────────────────────────────
        self._camera_info: CameraInfo | None = None

        # ── CV Bridge ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── 发布者 ────────────────────────────────────────────────────────────
        self.pub_pc2     = self.create_publisher(PointCloud2,  "/sam3/bev/pointcloud", queue_size)
        self.pub_markers = self.create_publisher(MarkerArray,  "/sam3/bev/markers",    queue_size)
        self.pub_bev_img = self.create_publisher(Image,        "/sam3/bev/image",      queue_size)

        # ── 相机内参独立订阅 ──────────────────────────────────────────────────
        self.sub_info = self.create_subscription(
            CameraInfo, info_topic, self._camera_info_cb, 1
        )

        # ── 三路近似时间同步：深度图 + 彩色图 + 检测结果 ─────────────────────
        self._sub_depth  = Subscriber(self, Image,           depth_topic)
        self._sub_color  = Subscriber(self, Image,           color_topic)
        self._sub_det3d  = Subscriber(self, DetectionArray,  det3d_topic)

        self._sync = ApproximateTimeSynchronizer(
            [self._sub_depth, self._sub_color, self._sub_det3d],
            queue_size=queue_size,
            slop=sync_slop,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._sync_callback)

        self.get_logger().info("BEV 投影节点已启动，等待消息…")

    # ── 内参回调 ──────────────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo):
        if self._camera_info is None:
            self.get_logger().info(
                f"相机内参已接收: fx={msg.k[0]:.2f} fy={msg.k[4]:.2f} "
                f"cx={msg.k[2]:.2f} cy={msg.k[5]:.2f}"
            )
        self._camera_info = msg

    # ── 同步回调主体 ──────────────────────────────────────────────────────────
    def _sync_callback(
        self,
        depth_msg: Image,
        color_msg: Image,
        det3d_msg: DetectionArray,
    ):
        if self._camera_info is None:
            self.get_logger().warn("相机内参尚未就绪，跳过。", throttle_duration_sec=3.0)
            return

        K = self._camera_info.k
        fx, fy, ppx, ppy = K[0], K[4], K[2], K[5]

        # ── 解码深度图 ──────────────────────────────────────────────────────
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_raw = np.asarray(depth_raw, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"深度图解码失败: {e}")
            return

        H, W = depth_raw.shape[:2]
        depth_m = depth_raw * self.depth_scale   # float32, 单位米

        # ── 解码彩色图 ──────────────────────────────────────────────────────
        try:
            color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"彩色图解码失败: {e}")
            return

        # 彩色图可能与深度图分辨率不同，统一缩放至深度图尺寸
        if color_bgr.shape[:2] != (H, W):
            color_bgr = cv2.resize(color_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── 构建像素坐标网格（下采样） ───────────────────────────────────────
        v_idx = np.arange(0, H, self.step, dtype=np.int32)
        u_idx = np.arange(0, W, self.step, dtype=np.int32)
        uu, vv = np.meshgrid(u_idx, v_idx)          # shape (Hs, Ws)
        uu_flat = uu.ravel()
        vv_flat = vv.ravel()

        d_flat = depth_m[vv_flat, uu_flat]           # 深度（米）

        # 过滤无效深度
        valid = (d_flat > 0) & np.isfinite(d_flat) & (d_flat <= self.max_depth)
        uu_v = uu_flat[valid]
        vv_v = vv_flat[valid]
        d_v  = d_flat[valid]

        if len(d_v) == 0:
            return

        # ── 反投影至相机坐标 (X, Y, Z) ──────────────────────────────────────
        X = (uu_v - ppx) * d_v / fx
        Y = (vv_v - ppy) * d_v / fy
        Z = d_v

        # ── 原始 RGB 颜色 ─────────────────────────────────────────────────
        bgr_flat = color_bgr[vv_v, uu_v]            # (N, 3)
        r_orig = bgr_flat[:, 2].astype(np.uint8)
        g_orig = bgr_flat[:, 1].astype(np.uint8)
        b_orig = bgr_flat[:, 0].astype(np.uint8)

        r_out = r_orig.copy()
        g_out = g_orig.copy()
        b_out = b_orig.copy()

        # ── 按分割掩膜着色 ──────────────────────────────────────────────────
        # SAM3 掩膜坐标系：imgsz 推理缩放空间 → 原始深度图空间
        max_side = max(H, W)
        infer_scale = self.imgsz / max_side if max_side > self.imgsz else 1.0

        # 记录每个像素所属的类别 id（-1 = 无）
        class_map = np.full(len(uu_v), -1, dtype=np.int32)

        for det in det3d_msg.detections:
            mask_msg = det.mask
            if mask_msg is None or mask_msg.height == 0:
                continue

            # 重建掩膜（推理空间）
            mask_infer = _reconstruct_mask(
                mask_msg.contour, mask_msg.height, mask_msg.width
            )
            # 缩放到深度图空间
            mask_depth = cv2.resize(
                mask_infer, (W, H), interpolation=cv2.INTER_NEAREST
            )
            # 查询有效像素的掩膜值
            in_mask = mask_depth[vv_v, uu_v] > 0
            class_map[in_mask] = det.class_id

        if self.colorize_class:
            # 对有类别标签的像素按类着色，其余保留原始颜色（灰化处理）
            for cid in np.unique(class_map):
                if cid < 0:
                    continue
                sel = class_map == cid
                cr, cg, cb = _class_color_rgb(cid)
                # 与原色混合（0.6 类色 + 0.4 原色）
                r_out[sel] = np.clip(0.6 * cr + 0.4 * r_orig[sel].astype(float), 0, 255).astype(np.uint8)
                g_out[sel] = np.clip(0.6 * cg + 0.4 * g_orig[sel].astype(float), 0, 255).astype(np.uint8)
                b_out[sel] = np.clip(0.6 * cb + 0.4 * b_orig[sel].astype(float), 0, 255).astype(np.uint8)
            # 无类别区域灰化（降低亮度以突出检测区域）
            no_class = class_map < 0
            gray = (0.299 * r_orig[no_class] + 0.587 * g_orig[no_class] + 0.114 * b_orig[no_class]).astype(np.uint8)
            r_out[no_class] = (gray * 0.5).astype(np.uint8)
            g_out[no_class] = (gray * 0.5).astype(np.uint8)
            b_out[no_class] = (gray * 0.5).astype(np.uint8)

        header = Header(frame_id=self.frame_id, stamp=depth_msg.header.stamp)

        # ── 发布 PointCloud2 ─────────────────────────────────────────────────
        if self.pub_pc2.get_subscription_count() > 0:
            pc_msg = self._make_pointcloud2(header, X, Y, Z, r_out, g_out, b_out)
            self.pub_pc2.publish(pc_msg)

        # ── 发布 MarkerArray ─────────────────────────────────────────────────
        if self.pub_markers.get_subscription_count() > 0:
            marker_msg = self._make_markers(header, det3d_msg.detections)
            self.pub_markers.publish(marker_msg)

        # ── 发布 BEV 光栅图像 ─────────────────────────────────────────────────
        if self.pub_bev_img.get_subscription_count() > 0:
            bev_img = self._make_bev_image(X, Y, Z, r_out, g_out, b_out, det3d_msg.detections)
            bev_ros = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
            bev_ros.header = header
            self.pub_bev_img.publish(bev_ros)

    # ── PointCloud2 构建 ──────────────────────────────────────────────────────
    def _make_pointcloud2(
        self,
        header: Header,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
    ) -> PointCloud2:
        """构建带 RGB 颜色的 XYZ PointCloud2 消息（16字节/点）。"""
        N = len(X)
        rgb_float = _pack_rgb_float(R, G, B)

        # 构建结构化数组
        data = np.zeros(N, dtype=np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32),
        ]))
        data["x"]   = X.astype(np.float32)
        data["y"]   = Y.astype(np.float32)
        data["z"]   = Z.astype(np.float32)
        data["rgb"] = rgb_float

        fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header     = header
        msg.height     = 1
        msg.width      = N
        msg.fields     = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step   = 16 * N
        msg.is_dense   = True
        msg.data       = data.tobytes()
        return msg

    # ── MarkerArray 构建 ──────────────────────────────────────────────────────
    def _make_markers(self, header: Header, detections) -> MarkerArray:
        """为每个检测目标生成球体 + 文字标签 Marker。"""
        markers = MarkerArray()
        mid = 0

        for i, det in enumerate(detections):
            b3d = det.bbox3d
            if b3d is None or b3d.depth_m < 0:
                continue

            pos = b3d.position
            color = _class_color_rgba(det.class_id, alpha=0.85)

            # ── 球体 Marker ────────────────────────────────────────────────
            sphere = Marker()
            sphere.header   = header
            sphere.ns       = "sam3_bev_sphere"
            sphere.id       = mid; mid += 1
            sphere.type     = Marker.SPHERE
            sphere.action   = Marker.ADD
            sphere.pose.position = Point(x=pos.x, y=pos.y, z=pos.z)
            sphere.pose.orientation.w = 1.0
            sphere.scale    = Vector3(x=0.12, y=0.12, z=0.12)
            sphere.color    = color
            sphere.lifetime = self._marker_lifetime
            markers.markers.append(sphere)

            # ── 文字标签 Marker ────────────────────────────────────────────
            label_text = (
                f"{det.class_name} {det.score:.2f}\n"
                f"Z={b3d.depth_m:.2f}m\n"
                f"({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})"
            )
            text_m = Marker()
            text_m.header   = header
            text_m.ns       = "sam3_bev_text"
            text_m.id       = mid; mid += 1
            text_m.type     = Marker.TEXT_VIEW_FACING
            text_m.action   = Marker.ADD
            # 标签稍微偏上（Y 轴向下，负方向为上）
            text_m.pose.position  = Point(x=pos.x, y=pos.y - 0.15, z=pos.z)
            text_m.pose.orientation.w = 1.0
            text_m.scale.z  = 0.08          # 文字高度（米）
            text_m.color    = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            text_m.text     = label_text
            text_m.lifetime = self._marker_lifetime
            markers.markers.append(text_m)

            # ── BEV 地面投影圆圈（在 XZ 平面） ────────────────────────────
            circle = Marker()
            circle.header   = header
            circle.ns       = "sam3_bev_circle"
            circle.id       = mid; mid += 1
            circle.type     = Marker.CYLINDER
            circle.action   = Marker.ADD
            # 在地面（Y = 0，也可用目标 Y，但 BEV 通常投到 y=0）
            circle.pose.position  = Point(x=pos.x, y=0.0, z=pos.z)
            circle.pose.orientation.w = 1.0
            # 扁圆柱体模拟地面圆圈：直径 0.4m，高度 0.02m
            circle.scale    = Vector3(x=0.4, y=0.4, z=0.02)
            circle.color    = _class_color_rgba(det.class_id, alpha=0.4)
            circle.lifetime = self._marker_lifetime
            markers.markers.append(circle)

        return markers

    # ── BEV 光栅图像构建 ──────────────────────────────────────────────────────
    def _make_bev_image(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        detections,
    ) -> np.ndarray:
        """
        将三维点投影到 BEV（俯视）光栅图像。
        X → 水平（列方向），Z → 垂直（行方向，远处在底部）。
        返回 BGR uint8 图像。
        """
        rows = self.bev_rows
        cols = self.bev_cols
        res  = self.bev_res

        # 深色背景
        bev = np.zeros((rows, cols, 3), dtype=np.uint8)
        bev[:] = (30, 30, 30)  # 深灰

        # ── 绘制网格线（每 1m 一条） ──────────────────────────────────────
        grid_step_px = int(round(1.0 / res))
        for gx in range(0, cols, grid_step_px):
            cv2.line(bev, (gx, 0), (gx, rows - 1), (60, 60, 60), 1)
        for gz in range(0, rows, grid_step_px):
            cv2.line(bev, (0, gz), (cols - 1, gz), (60, 60, 60), 1)

        # 绘制中心垂直轴（相机正前方）
        cx_center = cols // 2
        cv2.line(bev, (cx_center, 0), (cx_center, rows - 1), (80, 80, 80), 1)
        # 绘制相机位置（图像顶部中心）
        cv2.circle(bev, (cx_center, 0), 5, (0, 200, 200), -1)
        cv2.putText(bev, "CAM", (cx_center - 18, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)

        # ── 投影点云到 BEV 像素坐标 ─────────────────────────────────────
        # col = (X + bev_x_range) / res，row = (bev_z_max - Z) / res
        col_px = np.round((X + self.bev_x_range) / res).astype(np.int32)
        row_px = np.round((self.bev_z_max - Z) / res).astype(np.int32)

        in_bounds = (
            (col_px >= 0) & (col_px < cols) &
            (row_px >= 0) & (row_px < rows)
        )
        col_px = col_px[in_bounds]
        row_px = row_px[in_bounds]
        r_valid = R[in_bounds]
        g_valid = G[in_bounds]
        b_valid = B[in_bounds]

        # 逐点绘制（近处点后绘，避免远处点覆盖近处点）
        # 按 Z 从大到小排序（远 → 近）
        z_sort = Z[in_bounds]
        order = np.argsort(-z_sort)
        bev[row_px[order], col_px[order]] = np.stack(
            [b_valid[order], g_valid[order], r_valid[order]], axis=1
        )

        # ── 叠加检测目标标注 ──────────────────────────────────────────────
        for det in detections:
            b3d = det.bbox3d
            if b3d is None or b3d.depth_m < 0:
                continue
            pos = b3d.position
            dc = _class_color_bgr(det.class_id)

            # BEV 中心点
            cx = int(round((pos.x + self.bev_x_range) / res))
            cz = int(round((self.bev_z_max - pos.z) / res))
            if not (0 <= cx < cols and 0 <= cz < rows):
                continue

            # 实心圆
            cv2.circle(bev, (cx, cz), 8, dc, -1)
            cv2.circle(bev, (cx, cz), 8, (255, 255, 255), 1)

            # 类名 + 深度文字
            label = f"{det.class_name[:10]} {b3d.depth_m:.1f}m"
            tx = min(cx + 10, cols - 1)
            ty = min(cz + 5,  rows - 1)
            # 文字背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = max(0, min(tx, cols - tw - 2))
            ty = max(th + 2, ty)
            cv2.rectangle(bev, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2),
                           (0, 0, 0), cv2.FILLED)
            cv2.putText(bev, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, dc, 1, cv2.LINE_AA)

        # ── 边框及坐标轴标注 ──────────────────────────────────────────────
        cv2.rectangle(bev, (0, 0), (cols - 1, rows - 1), (100, 100, 100), 1)
        # 距离刻度标注（Z 轴）
        for dist in np.arange(1.0, self.bev_z_max, 1.0):
            r_label = int(round((self.bev_z_max - dist) / res))
            if 0 <= r_label < rows:
                cv2.putText(bev, f"{dist:.0f}m",
                            (2, r_label - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
        # X 轴范围标注
        cv2.putText(bev, f"-{self.bev_x_range:.0f}m",
                    (2, rows - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)
        cv2.putText(bev, f"+{self.bev_x_range:.0f}m",
                    (cols - 32, rows - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)

        return bev


# ── 入口 ─────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = BEVProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点异常退出: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
