from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit
from jax import numpy as jnp
import rclpy
from rclpy.node import Node
from pathlib import Path

from hsa_control_interfaces.msg import PlanarSetpoint
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa


class PlanarBendingTrajectoryNode(Node):
    def __init__(self):
        super().__init__("planar_bending_trajectory_node")

        self.declare_parameter("setpoint_topic", "setpoint")
        self.setpoint_pub = self.create_publisher(
            PlanarSetpoint, self.get_parameter("setpoint_topic").value, 10
        )

        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_hsa_ns-1_nrs-2.dill"
        )
        (
            forward_kinematics_virtual_backbone_fn,
            self.forward_kinematics_end_effector_fn,
            _,
            self.inverse_kinematics_end_effector_fn,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.params = PARAMS_CONTROL.copy()

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(
            self.params["sigma_a_eq"]
        )

        self.declare_parameter("phi_max", self.params["phi_max"].mean().item())
        self.params["phi_max"] = self.get_parameter("phi_max").value * jnp.ones_like(
            self.params["phi_max"]
        )

        # for now, we just set a constant setpoint
        self.declare_parameter("kappa_b_des", 7)
        self.kappa_b_des = self.get_parameter("kappa_b_des").value  # rad/m

        # initial setpoint index
        self.setpoint_idx = 0

        self.declare_parameter("planning_frequency", 5)
        self.timer = self.create_timer(
            1 / self.get_parameter("planning_frequency").value, self.timer_callback
        )

    def timer_callback(self):
        q_des = jnp.array([self.kappa_b_des, 0.0, 0.2876])

        # compute desired end effector pose with forward kinematics
        chiee_des = self.forward_kinematics_end_effector_fn(self.params, q_des)

        self.get_logger().info(f"chiee_des: {chiee_des}")

        msg = PlanarSetpoint()
        msg.chiee_des.x = chiee_des[0].item()
        msg.chiee_des.y = chiee_des[1].item()
        msg.chiee_des.theta = chiee_des[2].item()
        msg.q_des.header.stamp = self.get_clock().now().to_msg()
        msg.q_des.kappa_b = q_des[0].item()
        msg.q_des.sigma_sh = q_des[1].item()
        msg.q_des.sigma_a = q_des[2].item()
        self.setpoint_pub.publish(msg)

        self.setpoint_idx += 1


def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa_trajectory_planner.")

    node = PlanarBendingTrajectoryNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
