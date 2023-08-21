# Planar HSA control launch file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os

RECORD_BAG = False  # Record data to rosbag file
BAG_PATH = "/home/mstoelzle/phd/rosbags"
LOG_LEVEL = "warn"


def generate_launch_description():
    launch_actions = [
        Node(
            package="hsa_sim",
            executable="planar_sim_node",
            name="simulation",
        ),
        Node(
            package="hsa_visualization",
            executable="planar_viz_node",
            name="visualization",
            parameters=[
                {
                    "rendering_frequency": 20.0,
                }
            ],
        ),
        TimerAction(
            period=25.0,  # delay start of control node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_brain_control",
                    executable="planar_hsa_brain_control_node",
                    name="brain_control",
                ),
            ],
        ),
        Node(
            package="openvibe_bridge",
            executable="stimulation_receiver_node",
            name="brain_signal_publisher",
            parameters=[{}],
        ),
    ]

    if RECORD_BAG:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
