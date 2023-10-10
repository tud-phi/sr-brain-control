# Planar HSA control launch file
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import numpy as np
import os

# datetime object containing current date and time
now = datetime.now()

RECORD = True  # Record data to rosbag file
BAG_PATH = f"/home/mstoelzle/phd/rosbags/rosbag2_{now.strftime('%Y%m%d_%H%M%S')}"
LOG_LEVEL = "warn"

BRAIN_SIGNAL_SOURCE = "openvibe"  # "openvibe" or "keyboard"

hsa_material = "fpu"
kappa_b_eq = 0.0
sigma_sh_eq = 0.0
sigma_a_eq = 1.0
controller_type = "basic_operational_space_pid"

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
    "phi_delta": np.pi / 250,  # step for each stimulation [rad]
    "phi_max": phi_max,
}
planning_params = common_params | {
    "planning_frequency": 0.025  # period of 40s between setpoints
}
viz_params = common_params | {
    "rendering_frequency": 20.0,
    "invert_colors": True
}

control_params = common_params | {
    "controller_type": controller_type,
    "setpoint_topic": "/waypoint",
}
if controller_type == "basic_operational_space_pid":
    control_params.update(
        {
            "Kp": 1.0e1,  # [rad/m]
            "Ki": 1.1e2,  # [rad/(ms)]
            "Kd": 2.5e-1,  # [rad s/m]
        }
    )
else:
    raise ValueError(f"Unknown controller type: {controller_type}")


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
        Node(
            package="hsa_planar_control",
            executable="model_based_control_node",
            name="model_based_control",
            parameters=[control_params],
        ),
        TimerAction(
            period=40.0,  # delay start of control node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_brain_control",
                    executable="planar_hsa_cartesian_brain_control_node",
                    name="brain_control",
                    parameters=[common_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
        TimerAction(
            period=10.0,  # delay start of setpoint generation node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_planar_control",
                    executable="random_setpoints_node",
                    name="random_setpoints_generator",
                    parameters=[planning_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
    ]

    if BRAIN_SIGNAL_SOURCE == "openvibe":
        launch_actions.append(
            Node(
                package="joylike_operation",
                executable="openvibe_stimulation_to_joy_node",
                name="openvibe_teleop",
                parameters=[{"brain_control_mode": "cartesian", "host": "145.94.234.212"}],
                arguments=["--ros-args", "--log-level", LOG_LEVEL],
            ),
        )
    elif BRAIN_SIGNAL_SOURCE == "keyboard":
        keyboard2joy_filepath = os.path.join(
            get_package_share_directory("joylike_operation"),
            "config",
            "keystroke2joy_cartesian.yaml",
        )
        launch_actions.extend([
            Node(
                package="keyboard",
                executable="keyboard",
                name="keyboard",
            ),
            Node(
                package="joylike_operation",
                executable="keyboard_to_joy_node",
                name="keyboard_teleop",
                parameters=[{"config_filepath": str(keyboard2joy_filepath)}],
                arguments=["--ros-args", "--log-level", LOG_LEVEL],
            ),
        ])
    else:
        raise ValueError(f"Unknown brain signal source: {BRAIN_SIGNAL_SOURCE}")

    if RECORD:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
