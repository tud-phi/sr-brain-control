# Planar HSA control launch file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import numpy as np
import os

RECORD_BAG = False  # Record data to rosbag file
BAG_PATH = "/home/mstoelzle/phd/rosbags"
LOG_LEVEL = "warn"

hsa_material = "fpu"
kappa_b_eq = 0.0
sigma_sh_eq = 0.0
sigma_a_eq = 1.0

if hsa_material == "fpu":
    phi_max = 200 / 180 * np.pi
elif hsa_material == "epu":
    phi_max = 270 / 180 * np.pi
else:
    raise ValueError(f"Unknown HSA material: {hsa_material}")

common_params = {
    "hsa_material": hsa_material,
    "kappa_b_eq": kappa_b_eq,
    "sigma_sh_eq": sigma_sh_eq,
    "sigma_a_eq": sigma_a_eq,
    "phi_max": phi_max,
}
viz_params = common_params | {
    "rendering_frequency": 20.0,
}


def generate_launch_description():
    launch_actions = [
        Node(
            package="hsa_sim",
            executable="planar_sim_node",
            name="simulation",
            parameters=[common_params],
        ),
        Node(
            package="hsa_visualization",
            executable="planar_viz_node",
            name="visualization",
            parameters=[viz_params],
        ),
        TimerAction(
            period=30.0,  # delay start of control node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_brain_control",
                    executable="planar_hsa_brain_control_node",
                    name="brain_control",
                    parameters=[common_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
        Node(
            package="openvibe_bridge",
            executable="stimulation_receiver_node",
            name="brain_signal_publisher",
            parameters=[{"host": "145.94.196.114"}],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_trajectory_planner",
            executable="planar_bending_trajectory_node",
            name="trajectory_planner",
            parameters=[common_params],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
    ]

    if RECORD_BAG:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
