"""
SAM3 ROS2 节点 Launch 文件
用法：
  ros2 launch sam3_ros2 sam3.launch.py          # 不启动 RViz2
  ros2 launch sam3_ros2 sam3.launch.py rviz:=true  # 同时启动 RViz2 BEV 视图

所有参数统一在 config/sam3_config.yaml 中配置。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("sam3_ros2")
    config    = os.path.join(pkg_share, "config", "sam3_config.yaml")
    rviz_cfg  = os.path.join(pkg_share, "rviz", "sam3_bev.rviz")

    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="是否同时启动 RViz2 BEV 可视化界面",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    sam3_node = Node(
        package="sam3_ros2",
        executable="sam3_node",
        name="sam3_node",
        output="screen",
        emulate_tty=True,
        parameters=[config],
        remappings=[
            # ("/sam3/image_annotated", "/my_robot/sam3/annotated"),
        ],
    )

    depth_estimator_node = Node(
        package="sam3_ros2",
        executable="depth_estimator_node",
        name="depth_estimator_node",
        output="screen",
        emulate_tty=True,
        parameters=[config],
    )

    bev_projector_node = Node(
        package="sam3_ros2",
        executable="bev_projector_node",
        name="bev_projector_node",
        output="screen",
        emulate_tty=True,
        parameters=[config],
    )

    return LaunchDescription([
        rviz_arg,
        sam3_node,
        depth_estimator_node,
        bev_projector_node,
        rviz_node,
    ])
