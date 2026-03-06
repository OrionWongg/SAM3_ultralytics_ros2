# SAM3 ROS2 节点

基于 [Ultralytics SAM3](https://docs.ultralytics.com/models/sam-3/) 的 ROS2 Humble 语义分割封装节点。

## 功能

| 话题 | 方向 | 类型 | 说明 |
|---|---|---|---|
| `/camera/camera/color/image_raw` | 订阅 | `sensor_msgs/Image` | 输入彩色图像 |
| `/sam3/image_annotated` | 发布 | `sensor_msgs/Image` | 带掩码标注的结果图像 |
| `/sam3/results` | 发布 | `std_msgs/String` | 检测结果 JSON |

## 依赖

```bash
# ROS2 依赖
sudo apt install ros-humble-cv-bridge

# Python 依赖
pip install -U ultralytics
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git
```

> **注意**：SAM3 权重 `sam3.pt` 需前往 [HuggingFace](https://huggingface.co/facebook/sam3) 申请访问并手动下载。

## 构建

```bash
# 将包复制到 ROS2 工作空间的 src 目录，或在 sam3_test 目录下初始化工作空间
cd ~/your_ros2_ws
cp -r /home/hkulab/sam3_test/sam3_ros2 src/
colcon build --packages-select sam3_ros2
source install/setup.bash
```

## 运行

### 方式一：Launch 文件（推荐）

```bash
# 使用默认配置
ros2 launch sam3_ros2 sam3.launch.py

# 覆盖部分参数
ros2 launch sam3_ros2 sam3.launch.py \
  model_path:=/path/to/sam3.pt \
  conf:=0.3 \
  skip_frames:=1
```

### 方式二：直接运行节点

```bash
ros2 run sam3_ros2 sam3_node \
  --ros-args \
  --params-file /path/to/sam3_ros2/config/sam3_config.yaml
```

### 方式三：运行时修改参数

```bash
# 修改置信度阈值
ros2 param set /sam3_node conf 0.4

# 查看所有参数
ros2 param list /sam3_node
ros2 param dump /sam3_node
```

## 配置文件

所有参数集中在 [`config/sam3_config.yaml`](config/sam3_config.yaml)：

```yaml
sam3_node:
  ros__parameters:
    model_path: "sam3.pt"       # 模型权重路径
    conf: 0.25                  # 置信度阈值
    half: true                  # FP16 推理
    text_prompts:               # 分割目标（英文名词短语）
      - "person"
      - "car"
    input_topic: "/camera/camera/color/image_raw"
    queue_size: 1
    skip_frames: 0              # 跳帧（0=不跳，1=处理50%帧）
```

## JSON 结果格式

`/sam3/results` 话题发布的 JSON 格式示例：

```json
{
  "header": {
    "stamp_sec": 1234567890,
    "stamp_nanosec": 123456789,
    "frame_id": "camera_color_optical_frame"
  },
  "text_prompts": ["person", "car"],
  "num_detections": 2,
  "detections": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "person",
      "conf": 0.8721,
      "bbox_xyxy": [120.5, 80.3, 320.1, 450.7],
      "mask_area": 15234
    }
  ]
}
```

## 性能说明

SAM3 在 H200 GPU 上约 30ms/帧（100+ 目标），实际性能取决于硬件：

| 硬件 | 建议 skip_frames |
|---|---|
| H100/H200 GPU | 0（全帧处理） |
| RTX 3090/4090 | 0～1 |
| RTX 3060 及以下 | 1～2 |
