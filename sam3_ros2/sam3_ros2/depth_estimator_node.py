#!/usr/bin/env python3
"""
SAM3 深度估计节点 (depth_estimator_node)

订阅：
  • /camera/camera/aligned_depth_to_color/image_raw   — 对齐后的深度图（16UC1，单位 mm）
  • /camera/camera/aligned_depth_to_color/camera_info — 相机内参
  • /sam3/detections                                   — SAM3 检测结果（sam3_msgs/DetectionArray）
  • /sam3/image_annotated                              — SAM3 标注彩色图（缓存，覆盖深度信息用）

发布：
  • /sam3/detections_3d     — 填充了三维位置信息的 sam3_msgs/DetectionArray
  • /sam3/image_annotated_3d — 叠加了深度/三维坐标标注的图像

每个检测目标的"几何中心"取其 bounding-box 中心点，
映射回原始深度图坐标后查询深度值并换算为米，
同时利用相机内参将其反投影为相机坐标系三维坐标 (X, Y, Z)。
"""

import copy

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from sam3_msgs.msg import BoundingBox3D, DetectionArray


class DepthEstimatorNode(Node):
    """将 SAM3 检测结果与对齐深度图结合，输出每个目标的深度及三维坐标。"""

    def __init__(self):
        super().__init__("depth_estimator_node")

        # ── 参数声明 ──────────────────────────────────────────────────────────
        self.declare_parameter(
            "depth_image_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="对齐深度图话题",
            ),
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera/aligned_depth_to_color/camera_info",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="深度相机内参话题",
            ),
        )
        self.declare_parameter(
            "sam3_detections_topic",
            "/sam3/detections",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="SAM3 检测结果话题（sam3_msgs/DetectionArray）",
            ),
        )
        self.declare_parameter(
            "output_topic",
            "/sam3/detections_3d",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="输出带深度信息的 DetectionArray 话题",
            ),
        )
        self.declare_parameter(
            "queue_size",
            10,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="话题队列大小",
            ),
        )
        self.declare_parameter(
            "sync_slop",
            0.1,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="时间同步容差（秒），深度图与 SAM3 结果之间允许的最大时间差",
            ),
        )
        self.declare_parameter(
            "imgsz",
            518,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="SAM3 节点使用的推理分辨率（需与 sam3_node 的 imgsz 一致）",
            ),
        )
        self.declare_parameter(
            "depth_scale",
            0.001,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="深度值换算系数，默认 0.001（将 mm → m，适用于 RealSense 16UC1）",
            ),
        )
        self.declare_parameter(
            "max_depth",
            10.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="有效深度最大值（米），超出此范围视为无效深度（0 或 NaN 也视为无效）",
            ),
        )
        self.declare_parameter(
            "sample_radius",
            3,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="在几何中心周围采样的像素半径，取有效深度中位数（0 表示仅取单点）",
            ),
        )
        self.declare_parameter(
            "frame_id",
            "camera_color_optical_frame",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="三维坐标参考系 frame_id",
            ),
        )
        self.declare_parameter(
            "annotated_image_topic",
            "/sam3/image_annotated",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="SAM3 标注彩色图输入话题（用于叠加深度信息）",
            ),
        )
        self.declare_parameter(
            "output_annotated_topic",
            "/sam3/image_annotated_3d",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="叠加深度信息后输出的标注图话题",
            ),
        )

        # ── 读取参数 ──────────────────────────────────────────────────────────
        depth_topic        = self.get_parameter("depth_image_topic").value
        info_topic         = self.get_parameter("camera_info_topic").value
        sam3_topic         = self.get_parameter("sam3_detections_topic").value
        output_topic       = self.get_parameter("output_topic").value
        queue_size         = int(self.get_parameter("queue_size").value)
        sync_slop          = float(self.get_parameter("sync_slop").value)
        self.imgsz         = int(self.get_parameter("imgsz").value)
        self.depth_scale   = float(self.get_parameter("depth_scale").value)
        self.max_depth     = float(self.get_parameter("max_depth").value)
        self.sample_radius = int(self.get_parameter("sample_radius").value)
        self.frame_id      = self.get_parameter("frame_id").value
        annotated_topic    = self.get_parameter("annotated_image_topic").value
        output_ann_topic   = self.get_parameter("output_annotated_topic").value

        self.get_logger().info(f"深度图话题    : {depth_topic}")
        self.get_logger().info(f"相机内参话题  : {info_topic}")
        self.get_logger().info(f"SAM3 检测话题 : {sam3_topic}")
        self.get_logger().info(f"输出话题      : {output_topic}")
        self.get_logger().info(f"标注图输入    : {annotated_topic}")
        self.get_logger().info(f"标注图输出    : {output_ann_topic}")
        self.get_logger().info(f"时间同步容差  : {sync_slop} s")
        self.get_logger().info(f"SAM3 推理分辨率: {self.imgsz}")
        self.get_logger().info(f"采样半径      : {self.sample_radius} px")
        self.get_logger().info(f"坐标系        : {self.frame_id}")

        # ── 相机内参缓存 ──────────────────────────────────────────────────────
        self._camera_info: CameraInfo | None = None

        # ── 标注图缓存（独立订阅，取最近一帧即可） ─────────────────────────────
        self._latest_annotated: Image | None = None

        # ── CV Bridge ────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── 发布者 ────────────────────────────────────────────────────────────
        self.pub_detections_3d = self.create_publisher(
            DetectionArray,
            output_topic,
            queue_size,
        )
        self.pub_annotated_3d = self.create_publisher(
            Image,
            output_ann_topic,
            queue_size,
        )

        # ── 相机内参订阅（独立订阅，通常只需接收一次） ────────────────────────
        self.sub_info = self.create_subscription(
            CameraInfo,
            info_topic,
            self._camera_info_callback,
            1,
        )

        # ── 标注图独立订阅（缓存最新帧，与深度不做严格同步） ──────────────────
        self.sub_annotated = self.create_subscription(
            Image,
            annotated_topic,
            lambda msg: setattr(self, "_latest_annotated", msg),
            queue_size,
        )

        # ── 使用 message_filters 对深度图与 DetectionArray 做近似时间同步 ─────
        self._sub_depth = Subscriber(self, Image, depth_topic)
        self._sub_sam3  = Subscriber(self, DetectionArray, sam3_topic)

        self._sync = ApproximateTimeSynchronizer(
            [self._sub_depth, self._sub_sam3],
            queue_size=queue_size,
            slop=sync_slop,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._sync_callback)

        self.get_logger().info("深度估计节点已启动，等待消息…")

    # ── 相机内参回调 ──────────────────────────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        if self._camera_info is None:
            self.get_logger().info(
                f"相机内参已接收: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
            )
        self._camera_info = msg

    # ── 同步回调 ──────────────────────────────────────────────────────────────
    def _sync_callback(self, depth_msg: Image, sam3_msg: DetectionArray):
        """深度图 + sam3_msgs/DetectionArray 时间同步后触发。"""
        if not sam3_msg.detections:
            return

        # 解析深度图
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            depth_image = np.array(depth_image, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"深度图转换失败: {e}")
            return

        H_depth, W_depth = depth_image.shape[:2]

        # ── 计算缩放因子 ───────────────────────────────────────────────────
        # SAM3 节点在推理前将图像按 imgsz 等比缩放；bbox 在缩放后空间。
        # 深度图分辨率为原始分辨率，需将 bbox 中心点映射回原始坐标。
        max_side = max(H_depth, W_depth)
        scale = self.imgsz / max_side if max_side > self.imgsz else 1.0

        # 相机内参
        if self._camera_info is not None:
            K = self._camera_info.k  # 1×9 row-major
            fx, fy = K[0], K[4]
            ppx, ppy = K[2], K[5]
            has_intrinsics = True
        else:
            has_intrinsics = False
            self.get_logger().warn(
                "相机内参尚未接收，三维坐标将不可用。",
                throttle_duration_sec=5.0,
            )

        # 构建输出消息（深拷贝原始检测，填充 bbox3d）
        out_msg = DetectionArray()
        out_msg.header = sam3_msg.header
        out_msg.text_prompts = list(sam3_msg.text_prompts)

        for det in sam3_msg.detections:
            new_det = copy.deepcopy(det)

            # bbox 中心点（缩放后坐标）→ 原始深度图像素坐标
            cx_px = int(round((det.bbox.x1 + det.bbox.x2) / 2.0 / scale))
            cy_px = int(round((det.bbox.y1 + det.bbox.y2) / 2.0 / scale))
            cx_px = max(0, min(cx_px, W_depth - 1))
            cy_px = max(0, min(cy_px, H_depth - 1))

            depth_m = self._sample_depth(depth_image, cx_px, cy_px, H_depth, W_depth)

            # 填充 bbox3d
            bbox3d = BoundingBox3D()
            bbox3d.frame_id = self.frame_id
            bbox3d.depth_m = depth_m if depth_m is not None else -1.0

            if has_intrinsics and depth_m is not None:
                X = (cx_px - ppx) * depth_m / fx
                Y = (cy_px - ppy) * depth_m / fy
                Z = depth_m
                bbox3d.position = Point(
                    x=round(X, 4),
                    y=round(Y, 4),
                    z=round(Z, 4),
                )

            new_det.bbox3d = bbox3d
            out_msg.detections.append(new_det)

        self.pub_detections_3d.publish(out_msg)

        # ── 标注图叠加深度信息并发布 ──────────────────────────────────────────
        if (
            self._latest_annotated is not None
            and self.pub_annotated_3d.get_subscription_count() > 0
        ):
            try:
                ann_bgr = self.bridge.imgmsg_to_cv2(
                    self._latest_annotated, desired_encoding="bgr8"
                )
                self._draw_depth_overlay(ann_bgr, out_msg.detections)
                ann_out = self.bridge.cv2_to_imgmsg(ann_bgr, encoding="bgr8")
                ann_out.header = out_msg.header
                self.pub_annotated_3d.publish(ann_out)
            except Exception as e:
                self.get_logger().error(f"标注图深度叠加失败: {e}")

        # 控制台摘要（限速避免刷屏）
        summary = ", ".join(
            f"{d.class_name}({d.score:.2f})"
            + (f"@{d.bbox3d.depth_m:.3f}m" if d.bbox3d.depth_m >= 0.0 else "@N/A")
            for d in out_msg.detections
        )
        self.get_logger().info(
            f"深度估计: [{summary}]",
            throttle_duration_sec=1.0,
        )

    # ── 深度信息叠加 ──────────────────────────────────────────────────────────
    def _draw_depth_overlay(self, img: np.ndarray, detections) -> None:
        """
        在标注图上为每个目标叠加深度及三维坐标信息。
        坐标系：bbox 的 x1/y1/x2/y2 与标注图像素坐标一致（均为推理分辨率空间）。
        """
        for det in detections:
            b3d = det.bbox3d
            if b3d.depth_m < 0.0:
                depth_txt = "Z: N/A"
                xyz_txt   = ""
            else:
                depth_txt = f"Z: {b3d.depth_m:.2f}m"
                p = b3d.position
                xyz_txt = f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f})"

            # bbox 中心（在推理图坐标系）
            cx = int((det.bbox.x1 + det.bbox.x2) / 2.0)
            cy = int((det.bbox.y1 + det.bbox.y2) / 2.0)

            # 安全钳制
            H, W = img.shape[:2]
            cx = max(0, min(cx, W - 1))
            cy = max(0, min(cy, H - 1))

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness  = 1
            pad        = 4

            lines = [depth_txt] + ([xyz_txt] if xyz_txt else [])
            (tw, th), baseline = cv2.getTextSize(
                max(lines, key=len), font, font_scale, thickness
            )
            line_h = th + baseline + 2

            # 背景矩形起始点（显示在 bbox 中心下方）
            rx = cx - tw // 2 - pad
            ry = cy + pad
            rect_w = tw + pad * 2
            rect_h = line_h * len(lines) + pad

            # 确保矩形不超出图像边界
            rx = max(0, min(rx, W - rect_w))
            ry = max(0, min(ry, H - rect_h))

            # 半透明背景
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (rx, ry),
                (rx + rect_w, ry + rect_h),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

            # 文字（白色）
            for k, line in enumerate(lines):
                ty = ry + (k + 1) * line_h - baseline
                cv2.putText(img, line, (rx + pad, ty), font, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)

            # 小圆点标记中心
            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
            cv2.circle(img, (cx, cy), 4, (0, 0, 0),     1)

    # ── 深度采样 ──────────────────────────────────────────────────────────────
    def _sample_depth(
        self,
        depth_image: np.ndarray,
        cx: int,
        cy: int,
        H: int,
        W: int,
    ) -> float | None:
        """
        在 (cx, cy) 周围 sample_radius 个像素范围内采集有效深度值，
        取中位数后换算为米；若无有效点则返回 None。
        """
        r = self.sample_radius
        if r <= 0:
            raw = depth_image[cy, cx]
            if raw <= 0:
                return None
            d = float(raw) * self.depth_scale
            return d if d <= self.max_depth else None

        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)

        patch = depth_image[y0:y1, x0:x1].flatten()
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if len(valid) == 0:
            return None

        d = float(np.median(valid)) * self.depth_scale
        return round(d, 4) if d <= self.max_depth else None


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
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
