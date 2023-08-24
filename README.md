# sr-brain-control

Brain-control of soft robots

## Usage

### Launch ROS2 nodes

```bash
ros2 launch ./launch/planar_hsa_sim_brain_control_launch.py
```

### Manually send brain signals

```bash
ros2 topic pub /brain_signal std_msgs/Int32 'data: 1.0' -1
```
