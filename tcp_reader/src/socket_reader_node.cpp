// Original ROS1 implementation: 
// https://github.com/Tonnius/bci_ssvep_openvibe_ros/tree/master/ROS/openvibe_to_ros_tcp
// which is again based on http://www.linuxhowtos.org/C_C++/socket.htm

#include "rclcpp/rclcpp.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include "std_msgs/msg/string.hpp"
#include <sstream>
#define MESSAGE_FREQ 100

void error(const char *msg) {
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::Node node("socket_reader_node");

  auto publisher = node.create_publisher<std_msgs::msg::String>("chatter", 10);

  rclcpp::Rate loop_rate(MESSAGE_FREQ); // Set rate as defined in the macro MESSAGE_FREQ

  int sockfd, portno, n;
  struct sockaddr_in serv_addr;
  struct hostent *server;
  char buffer[256];

  if (argc < 3) {
      fprintf(stderr,"Usage: $ ros2 run tcp_reader socket_reader_node <hostname> <port>\n");
      exit(0);
  }
  
  // Open socket
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) 
      error("ERROR opening socket");

  portno = atoi(argv[2]);
  server = gethostbyname(argv[1]);
  if (server == NULL) {
      fprintf(stderr,"ERROR, no such host\n");
      exit(0);
  }

  // Configure socket parameters
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;

  bcopy((char *)server->h_addr, 
        (char *)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(portno);

  // Connect to socket
  if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
      error("ERROR connecting");

  std_msgs::msg::String message;
  std::stringstream ss;

  while(rclcpp::ok()) {
        ss.str(std::string()); // Clear contents of string stream
        bzero(buffer, 256);
        n = read(sockfd,buffer, 255); // Read msg from buffer
        if (n < 0) 
            error("ERROR reading from socket");

        ss << static_cast<int>(buffer[0]);
        message.data = ss.str(); 

        RCLCPP_INFO(node.get_logger(), "I heard: %s", message.data.c_str());
        publisher->publish(message); // Publish msg to chatter

      // ros::spinOnce();
  }

  close(sockfd);
	return 0;
}