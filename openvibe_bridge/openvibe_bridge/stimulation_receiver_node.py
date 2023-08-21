#!/usr/bin/env python3

import socket
import rclpy
from std_msgs.msg import Int32  # Import the appropriate message type


def decode_stimulation(byte_data):
    # Decode the first byte to determine the stimulation type
    stimulation_type = int(byte_data[0])

    # Return the decoded stimulation type
    return stimulation_type


def main():
    rclpy.init()

    node = rclpy.create_node("stimulation_receiver_node")
    publisher = node.create_publisher(Int32, "brain_signal", 10)

    node.declare_parameter("host", "localhost")
    host = node.get_parameter("host").value
    node.declare_parameter("port", 5678)
    port = node.get_parameter("port").value

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        print(f"Listening on {host}:{port}")

        while rclpy.ok():
            tcp_data = s.recv(8)
            if not tcp_data:
                break

            node.get_logger().info(f"Received raw tcp data: {tcp_data}")

            # Decode the received data
            stimulation_type = decode_stimulation(tcp_data)
            node.get_logger().info(
                f"Decoded msg to stimulation type: {stimulation_type}"
            )

            brain_signal = 0
            # map the stimulation type to the brain signal {-1, 0, 1}
            if stimulation_type == 1:
                # bending to the left
                brain_signal = 1
            elif stimulation_type == 2:
                # bending to the right
                brain_signal = -1
            else:
                raise ValueError(f"Unknown stimulation type {stimulation_type}")

            # Create an instance of your custom message
            # Assign received data to the message field
            msg = Int32(data=brain_signal)

            publisher.publish(msg)
            node.get_logger().info(f"Published msg: {msg}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
