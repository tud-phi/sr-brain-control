from jax import Array, jit, vmap
import jax.numpy as jnp
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Joy

from hsa_control_interfaces.msg import PlanarSetpoint


class PlanarHsaCartesianBrainControlNode(Node):
    def __init__(self):
        super().__init__("planar_hsa_cartesian_brain_control_node")

        # change of position in cartesian space at each time step in unit [m]
        self.declare_parameter("cartesian_delta", 1e-4)  # 0.1 mm
        self.cartesian_delta = self.get_parameter("cartesian_delta").value

        # publisher of waypoints planned by the brain / user
        self.declare_parameter("waypoint_topic", "waypoint")
        self.waypoint_pub = self.create_publisher(
            PlanarSetpoint, self.get_parameter("waypoint_topic").value, 10
        )

        # if the robot is platform-down, the coordinates are inverted and with that we also need to invert the brain signals
        self.declare_parameter("invert_brain_signals", True)
        self.invert_brain_signals = self.get_parameter("invert_brain_signals").value

        # intial waypoint position
        self.declare_parameter("pee_y0", 0.0) # [m]
        # end-effector position desired by the brain / user
        self.pee_des = jnp.array([0.0, self.get_parameter("pee_y0").value])

        self.declare_parameter("brain_signal_topic", "brain_signal")
        self.brain_signal_sub = self.create_subscription(
            Joy,
            self.get_parameter("brain_signal_topic").value,
            self.brain_signal_callback,
            10,
        )

    def brain_signal_callback(self, msg: Joy):
        brain_signal = jnp.array(msg.axes)
        self.get_logger().info(f"Received brain signal: {brain_signal}")

        # compute the position of the next waypoint
        if self.invert_brain_signals:
            self.pee_des = self.pee_des - self.cartesian_delta * brain_signal
        else:
            self.pee_des = self.pee_des + self.cartesian_delta * brain_signal

        # publish waypoint
        msg = PlanarSetpoint()
        # we don't specify the orientation of the end-effector
        # so we just set a dummy value
        msg.chiee_des = Pose2D(x=self.pee_des[0].item(), y=self.pee_des[1].item(), theta=0.0)
        self.waypoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from planar_hsa_cartesian_brain_control_node.")

    node = PlanarHsaCartesianBrainControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
