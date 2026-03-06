#!/usr/bin/env python3
"""
SAM3 ROS2 节点
订阅相机彩色图像话题，使用 SAM3SemanticPredictor 进行文本提示分割，
发布标注结果图像和 sam3_msgs/DetectionArray 检测结果。
"""

import queue
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import Image

from sam3_msgs.msg import (
    BoundingBox2D,
    BoundingBox3D,
    Detection,
    DetectionArray,
    Mask,
    Point2D,
)


class SAM3Node(Node):
    """SAM3 语义分割 ROS2 节点。"""

    def __init__(self):
        super().__init__("sam3_node")

        # ── 声明参数 ──────────────────────────────────────────────────────────
        self.declare_parameter(
            "model_path",
            "sam3.pt",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="SAM3 模型权重文件路径",
            ),
        )
        self.declare_parameter(
            "conf",
            0.25,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="检测置信度阈值",
            ),
        )
        self.declare_parameter(
            "half",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="是否使用 FP16 半精度推理",
            ),
        )
        self.declare_parameter(
            "text_prompts",
            ["person", "car"],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="文本分割提示词列表（名词短语）",
            ),
        )
        self.declare_parameter(
            "input_topic",
            "/camera/camera/color/image_raw",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="输入图像话题",
            ),
        )
        self.declare_parameter(
            "queue_size",
            1,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="话题队列大小",
            ),
        )
        self.declare_parameter(
            "skip_frames",
            0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="连续跳帧数（减少推理压力，0 表示不跳帧）",
            ),
        )
        self.declare_parameter(
            "imgsz",
            644,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="推理输入分辨率（长边像素数，须为14的倍数），降低可显著提速，建议 630/644",
            ),
        )
        self.declare_parameter(
            "device",
            "cuda:0",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="推理设备，如 cuda:0 / cuda:1 / cpu",
            ),
        )
        self.declare_parameter(
            "warmup_iters",
            2,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="模型加载后预热推理次数，消除首帧延迟",
            ),
        )
        self.declare_parameter(
            "publish_annotated",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="是否发布标注图像（关闭可节省 result.plot() 开销）",
            ),
        )

        # ── 读取参数 ──────────────────────────────────────────────────────────
        self.model_path      = self.get_parameter("model_path").value
        self.conf            = float(self.get_parameter("conf").value)
        self.half            = bool(self.get_parameter("half").value)
        self.text_prompts    = list(self.get_parameter("text_prompts").value)
        input_topic          = self.get_parameter("input_topic").value
        queue_size           = int(self.get_parameter("queue_size").value)
        self.skip_frames     = int(self.get_parameter("skip_frames").value)
        self.imgsz           = int(self.get_parameter("imgsz").value)
        self.device          = self.get_parameter("device").value
        self.warmup_iters    = int(self.get_parameter("warmup_iters").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self.get_logger().info(f"模型路径     : {self.model_path}")
        self.get_logger().info(f"置信度阈值   : {self.conf}")
        self.get_logger().info(f"FP16 模式    : {self.half}")
        self.get_logger().info(f"文本提示词   : {self.text_prompts}")
        self.get_logger().info(f"输入话题     : {input_topic}")
        self.get_logger().info(f"跳帧数       : {self.skip_frames}")
        self.get_logger().info(f"推理分辨率   : {self.imgsz}")
        self.get_logger().info(f"推理设备     : {self.device}")
        self.get_logger().info(f"预热次数     : {self.warmup_iters}")
        self.get_logger().info(f"发布标注图   : {self.publish_annotated}")

        # ── CV Bridge ────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── 发布者 ────────────────────────────────────────────────────────────
        self.pub_annotated = self.create_publisher(
            Image,
            "/sam3/image_annotated",
            queue_size,
        )
        self.pub_results = self.create_publisher(
            DetectionArray,
            "/sam3/detections",
            queue_size,
        )

        # ── 状态变量 ──────────────────────────────────────────────────────────
        self._frame_count = 0
        self._inference_lock = threading.Lock()
        self._predictor = None
        self._predictor_ready = False

        # ── 异步标注发布队列（推理线程 → 发布线程，避免 plot() 阻塞推理） ─────
        self._annotate_queue: queue.Queue = queue.Queue(maxsize=2)
        self._annotate_thread = threading.Thread(
            target=self._annotate_worker, daemon=True
        )
        self._annotate_thread.start()

        # ── 后台加载模型 ──────────────────────────────────────────────────────
        load_thread = threading.Thread(target=self._load_model, daemon=True)
        load_thread.start()

        # ── 订阅者（等模型加载完再处理，但先建立订阅） ────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            queue_size,
        )

        self.get_logger().info("SAM3 节点已启动，正在后台加载模型…")

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    def _load_model(self):
        """在后台线程中初始化 SAM3SemanticPredictor，并执行预热推理。"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.conf,
                task="segment",
                mode="predict",
                model=self.model_path,
                half=self.half,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                save=False,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
            # 调用 setup_model() 提前触发模型权重加载
            predictor.setup_model(model=None)
            # setup_model 可能重置 args，强制嵌入 conf 确保生效
            predictor.args.conf = self.conf

            # ── 预热：用黑色哑帧跑几次，让 CUDA 完成 JIT 编译和内核缓存 ──────
            if self.warmup_iters > 0:
                self.get_logger().info(f"开始模型预热（{self.warmup_iters} 次）…")
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                with torch.no_grad():
                    for _ in range(self.warmup_iters):
                        predictor.set_image(dummy)
                        predictor(text=self.text_prompts)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.get_logger().info("模型预热完成。")

            with self._inference_lock:
                self._predictor = predictor
                self._predictor_ready = True

            self.get_logger().info("SAM3 模型加载完成，开始处理图像。")
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {e}")

    # ── 标注发布工作线程 ──────────────────────────────────────────────────────
    def _annotate_worker(self):
        """独立线程：消费 result，执行 plot() 并发布标注图像，不占用推理时间。"""
        while True:
            try:
                item = self._annotate_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:  # 关闭信号
                break
            result, header = item
            try:
                annotated_bgr = result.plot(labels=True, conf=True, masks=True)
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
                annotated_msg.header = header
                self.pub_annotated.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f"标注图像发布失败: {e}")

    # ── 图像回调 ──────────────────────────────────────────────────────────────
    def _image_callback(self, msg: Image):
        """接收 ROS 图像消息并执行 SAM3 推理。"""
        # 跳帧逻辑
        self._frame_count += 1
        if self.skip_frames > 0 and (self._frame_count % (self.skip_frames + 1)) != 1:
            return

        # 模型未就绪则跳过
        if not self._predictor_ready:
            self.get_logger().warn("模型尚未加载完成，跳过当前帧。", throttle_duration_sec=3.0)
            return

        # 若上一帧还在推理，非阻塞跳过（防止队列堆积）
        if not self._inference_lock.acquire(blocking=False):
            self.get_logger().warn("推理忙碌，跳过当前帧。", throttle_duration_sec=1.0)
            return

        try:
            self._run_inference(msg)
        except Exception as e:
            self.get_logger().error(f"推理出错: {e}")
        finally:
            self._inference_lock.release()

    # ── 推理主体 ──────────────────────────────────────────────────────────────
    def _run_inference(self, msg: Image):
        """执行 SAM3 推理并发布结果。"""
        # ROS Image → OpenCV BGR numpy
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # ── 按 imgsz 预缩放，减少编码器计算量 ──────────────────────────────
        h, w = cv_image.shape[:2]
        max_side = max(h, w)
        if max_side > self.imgsz:
            scale = self.imgsz / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        t0 = time.time()

        # SAM3 推理（torch.no_grad 禁用梯度追踪，节省显存和计算）
        predictor = self._predictor
        with torch.no_grad():
            predictor.set_image(cv_image)
            results = predictor(text=self.text_prompts, conf=self.conf)

        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(
            f"推理完成: {elapsed_ms:.1f} ms",
            throttle_duration_sec=1.0,
        )

        if not results:
            return

        result = results[0]

        # ── 发布标注图像（投递给异步线程，不阻塞当前推理） ─────────────────
        if self.publish_annotated and self.pub_annotated.get_subscription_count() > 0:
            try:
                self._annotate_queue.put_nowait((result, msg.header))
            except queue.Full:
                self.get_logger().warn("标注队列已满，跳过本帧标注。", throttle_duration_sec=1.0)

        # ── 发布 DetectionArray 结果（无条件发布，message_filters 订阅者不计入 subscription_count） ──
        det_array = self._build_detection_array(result, msg.header)
        self.pub_results.publish(det_array)

    # ── 构建 DetectionArray ──────────────────────────────────────────────────
    def _build_detection_array(self, result, header) -> DetectionArray:
        """将 Ultralytics Results 转为 sam3_msgs/DetectionArray 消息。"""
        msg = DetectionArray()
        msg.header = header
        msg.text_prompts = list(self.text_prompts)

        boxes = result.boxes   # 可能为 None
        masks = result.masks   # 可能为 None

        # result.names 可能是 dict {id: name} 或 list
        _names = result.names
        if isinstance(_names, dict):
            names = _names
        elif isinstance(_names, (list, tuple)):
            names = {i: n for i, n in enumerate(_names)}
        else:
            names = {}

        n_det = len(boxes) if boxes is not None else 0

        # 提前获取 mask tensor（统一 .cpu()，减少多次同步）
        mask_data = None
        mask_h, mask_w = 0, 0
        if masks is not None and len(masks) > 0:
            mask_data = masks.data.cpu().numpy()   # shape: (N, H, W) float32 [0,1]
            mask_h, mask_w = mask_data.shape[1], mask_data.shape[2]

        for i in range(n_det):
            box = boxes[i]
            det = Detection()
            det.class_id  = int(box.cls.item())
            det.class_name = names.get(det.class_id, "unknown")
            det.score     = round(float(box.conf.item()), 4)

            # 2D bbox (xyxy)
            xyxy = box.xyxy[0].tolist()
            det.bbox = BoundingBox2D(
                x1=round(xyxy[0], 2),
                y1=round(xyxy[1], 2),
                x2=round(xyxy[2], 2),
                y2=round(xyxy[3], 2),
            )

            # 3D bbox（此处留空，由 depth_estimator_node 填充）
            det.bbox3d = BoundingBox3D(depth_m=-1.0)

            # 分割掩码信息
            mask_msg = Mask(height=mask_h, width=mask_w)
            if mask_data is not None and i < len(mask_data):
                binary = (mask_data[i] > 0.5).astype(np.uint8)
                mask_msg.area = int(binary.sum())
                # 提取轮廓边界点
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # 取面积最大的轮廓
                    largest = max(contours, key=cv2.contourArea)
                    mask_msg.contour = [
                        Point2D(x=float(pt[0][0]), y=float(pt[0][1]))
                        for pt in largest
                    ]
            det.mask = mask_msg

            msg.detections.append(det)

        return msg


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SAM3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点异常退出: {e}")
    finally:
        # 发送关闭信号给标注线程
        try:
            node._annotate_queue.put_nowait(None)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
