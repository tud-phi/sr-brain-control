import numpy as np
import rclpy
from rclpy.node import Node

from example_interfaces.msg import Float64MultiArray
from sensor_msgs.msg import Joy


class PlanarHsaBrainControlNode(Node):
    def __init__(self):
        super().__init__("planar_hsa_brain_control_node")

        # offset (neural) control input [rad]
        self.declare_parameter("phi_offset", np.pi / 2)
        self.phi_offset = self.get_parameter("phi_offset").value

        # change of phi at each time step in unit [rad]
        self.declare_parameter("phi_delta", np.pi / 25)
        self.phi_delta = self.get_parameter("phi_delta").value

        # maximum magnitude of control input [rad]
        self.declare_parameter("phi_max", np.pi)
        self.phi_max = self.get_parameter("phi_max").value

        # publisher of control input
        self.declare_parameter("control_input_topic", "control_input")
        self.control_input_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("control_input_topic").value, 10
        )

        # initialize control input
        self.phi = self.phi_offset * np.ones((2,))
        # publish initial control input
        self.control_input_pub.publish(Float64MultiArray(data=self.phi.tolist()))

        self.declare_parameter("brain_signal_topic", "brain_signal")
        self.brain_signal_sub = self.create_subscription(
            Joy,
            self.get_parameter("brain_signal_topic").value,
            self.brain_signal_callback,
            10,
        )

    def brain_signal_callback(self, msg: Joy):
        # brain signal will be Int32 with values {-1, 0, 1}
        brain_signal = np.array(msg.axes).item()
        self.get_logger().info("Received brain signal: %d" % brain_signal)

        # calculate control input
        self.phi = self.phi + self.phi_delta * brain_signal * np.array([1.0, -1.0])

        # saturate control input to [0.0, phi_max]
        self.phi = np.clip(
            self.phi, np.zeros_like(self.phi), self.phi_max * np.ones_like(self.phi)
        )

        # publish control input
        phi_msg = Float64MultiArray(data=self.phi.tolist())
        self.control_input_pub.publish(phi_msg)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from hsa_brain_control.")

    node = PlanarHsaBrainControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
