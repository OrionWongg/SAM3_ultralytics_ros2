# SAM3 Ultralytics ROS2

基于 [Ultralytics SAM3](https://docs.ultralytics.com/models/sam-3/) 的 ROS2 Humble 语义分割套件，支持从 2D 图像检测到 3D 定位、BEV 俯视图的完整感知管线。

---

## 目录

- [架构概览](#架构概览)
- [依赖安装](#依赖安装)
- [模型下载](#模型下载)
- [工作空间构建](#工作空间构建)
- [运行](#运行)
- [话题速查](#话题速查)
- [关键参数说明](#关键参数说明)
- [自定义消息](#自定义消息)

---

## 架构概览

```
Camera (RealSense)
  │
  ├─ /camera/.../color/image_raw ──────────────► sam3_node
  │                                                  │
  │                                         /sam3/detections (DetectionArray)
  │                                         /sam3/image_annotated (Image)
  │
  ├─ /camera/.../aligned_depth_to_color/image_raw ─► depth_estimator_node
  │   (+ /sam3/detections)                               │
  │                                              /sam3/detections_3d (DetectionArray)
  │                                              /sam3/image_annotated_3d (Image)
  │
  └─ (depth + color + detections_3d) ──────────► bev_projector_node
                                                       │
                                              /sam3/bev/pointcloud (PointCloud2)
                                              /sam3/bev/markers    (MarkerArray)
                                              /sam3/bev/image      (Image)
```

| 包 | 说明 |
|---|---|
| `sam3_msgs` | 自定义消息定义（`DetectionArray`、`Detection`、`Mask` 等） |
| `sam3_ros2` | 三个 ROS2 节点 + Launch 文件 + 配置文件 |

---

## 依赖安装

### 系统依赖

```bash
# ROS2 Humble
sudo apt install ros-humble-cv-bridge ros-humble-rviz2
```

### Python 依赖

```bash
pip install -U ultralytics
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git
```

---

## 模型下载

SAM3 权重文件需前往 [HuggingFace](https://huggingface.co/facebook/sam3) 申请访问后手动下载 `sam3.pt`，
将其放置到任意路径并在配置文件中指定：

```yaml
# config/sam3_config.yaml
sam3_node:
  ros__parameters:
    model_path: "/path/to/sam3.pt"
```

---

## 工作空间构建

```bash
cd ~/sam3_ws
colcon build --packages-select sam3_msgs sam3_ros2
source install/setup.bash
```

---

## 运行

### 方式一：Launch 文件（推荐）

```bash
# 启动三个节点
ros2 launch sam3_ros2 sam3.launch.py

# 同时打开 RViz2 BEV 可视化
ros2 launch sam3_ros2 sam3.launch.py rviz:=true
```

### 方式二：单独运行节点

```bash
# 仅运行 SAM3 分割节点
ros2 run sam3_ros2 sam3_node \
  --ros-args --params-file ~/sam3_ws/src/sam3_ros2/config/sam3_config.yaml

# 仅运行深度估计节点
ros2 run sam3_ros2 depth_estimator_node \
  --ros-args --params-file ~/sam3_ws/src/sam3_ros2/config/sam3_config.yaml

# 仅运行 BEV 投影节点
ros2 run sam3_ros2 bev_projector_node \
  --ros-args --params-file ~/sam3_ws/src/sam3_ros2/config/sam3_config.yaml
```

### 方式三：运行时动态修改参数

```bash
# 修改置信度阈值
ros2 param set /sam3_node conf 0.4

# 修改分割提示词
ros2 param set /sam3_node text_prompts '["person", "car", "dog"]'

# 查看所有参数
ros2 param list /sam3_node
ros2 param dump /sam3_node
```

---

## 话题速查

### sam3_node

| 话题 | 方向 | 类型 | 说明 |
|---|---|---|---|
| `/camera/camera/color/image_raw` | 订阅 | `sensor_msgs/Image` | 输入彩色图像（可通过 `input_topic` 参数修改） |
| `/sam3/detections` | 发布 | `sam3_msgs/DetectionArray` | 2D 检测结果（含掩码） |
| `/sam3/image_annotated` | 发布 | `sensor_msgs/Image` | 带掩码标注的可视化图像 |

### depth_estimator_node

| 话题 | 方向 | 类型 | 说明 |
|---|---|---|---|
| `/camera/camera/aligned_depth_to_color/image_raw` | 订阅 | `sensor_msgs/Image` | 对齐深度图（16UC1，mm） |
| `/camera/camera/aligned_depth_to_color/camera_info` | 订阅 | `sensor_msgs/CameraInfo` | 深度相机内参 |
| `/sam3/detections` | 订阅 | `sam3_msgs/DetectionArray` | 来自 sam3_node 的 2D 检测 |
| `/sam3/detections_3d` | 发布 | `sam3_msgs/DetectionArray` | 附加 3D 坐标的检测结果 |
| `/sam3/image_annotated_3d` | 发布 | `sensor_msgs/Image` | 叠加深度信息的标注图像 |

### bev_projector_node

| 话题 | 方向 | 类型 | 说明 |
|---|---|---|---|
| `/camera/camera/aligned_depth_to_color/image_raw` | 订阅 | `sensor_msgs/Image` | 对齐深度图 |
| `/camera/camera/color/image_raw` | 订阅 | `sensor_msgs/Image` | 彩色图像 |
| `/camera/camera/aligned_depth_to_color/camera_info` | 订阅 | `sensor_msgs/CameraInfo` | 相机内参 |
| `/sam3/detections_3d` | 订阅 | `sam3_msgs/DetectionArray` | 3D 检测结果 |
| `/sam3/bev/pointcloud` | 发布 | `sensor_msgs/PointCloud2` | 按类别着色点云 |
| `/sam3/bev/markers` | 发布 | `visualization_msgs/MarkerArray` | 3D 检测框标注 |
| `/sam3/bev/image` | 发布 | `sensor_msgs/Image` | BEV 俯视光栅图 |

---

## 关键参数说明

所有参数均在 [`sam3_ros2/config/sam3_config.yaml`](sam3_ros2/config/sam3_config.yaml) 中配置，Launch 文件会自动加载。

| 参数 | 节点 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | sam3_node | `"sam3.pt"` | SAM3 权重文件路径 |
| `conf` | sam3_node | `0.7` | 检测置信度阈值 |
| `half` | sam3_node | `true` | 是否使用 FP16 半精度推理 |
| `text_prompts` | sam3_node | `["ground", ...]` | 分割目标（英文名词短语） |
| `input_topic` | sam3_node | `/camera/.../color/image_raw` | 输入图像话题 |
| `skip_frames` | sam3_node | `0` | 跳帧数（0=不跳，1=处理50%帧） |
| `imgsz` | sam3_node | `518` | 推理分辨率（需为14的倍数） |
| `device` | sam3_node | `"cuda:0"` | 推理设备 |
| `publish_annotated` | sam3_node | `true` | 是否发布可视化标注图像 |
| `depth_scale` | depth_estimator_node | `0.001` | 深度原始值到米的换算系数 |
| `max_depth` | depth_estimator_node | `10.0` | 有效深度上限（米） |
| `sample_radius` | depth_estimator_node | `3` | 深度采样半径（像素） |
| `sync_slop` | depth_estimator_node | `0.5` | 时间同步容差（秒） |
| `bev_x_range` | bev_projector_node | `5.0` | BEV 图 X 轴单侧范围（米） |
| `bev_z_max` | bev_projector_node | `10.0` | BEV 图最大深度（米） |
| `bev_resolution` | bev_projector_node | `0.02` | 每像素物理距离（米） |

### 性能建议（`skip_frames`）

| 硬件 | 建议值 |
|---|---|
| H100 / H200 | 0（全帧） |
| RTX 3090 / 4090 | 0 ~ 1 |
| RTX 3060 及以下 | 1 ~ 2 |

---

## 自定义消息

`sam3_msgs` 包提供以下消息类型：

| 消息 | 说明 |
|---|---|
| `DetectionArray` | 检测结果数组，含 header 和 `Detection[]` |
| `Detection` | 单个检测：类别 ID/名称、置信度、2D/3D 边界框、掩码 |
| `BoundingBox2D` | 2D 像素坐标边界框 |
| `BoundingBox3D` | 3D 坐标边界框（中心 + 尺寸） |
| `Mask` | 实例分割掩码（编码图像数据） |
| `Point2D` | 2D 点坐标 |
