#!/usr/bin/env python3

import socket
import rclpy
from std_msgs.msg import String  # Import the appropriate message type


def main():
    rclpy.init()

    node = rclpy.create_node("socket_reader_node")
    publisher = node.create_publisher(String, "socket_data", 10)

    node.declare_parameter("host", "localhost")
    host = node.get_parameter("host").value
    node.declare_parameter("port", 5678)
    port = node.get_parameter("port").value

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()

        print(f"Listening on {host}:{port}")
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        while rclpy.ok():
            data = conn.recv(1024).decode("utf-8")
            if not data:
                break

            node.get_logger().info(f"Received: {data}")

            # Create an instance of your custom message
            # Assign received data to the message field
            msg = String(data=data)

            publisher.publish(msg)
            node.get_logger().info(f"Published: {msg}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
