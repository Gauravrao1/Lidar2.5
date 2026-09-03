"""
ROS2 nodes for the segmentation and grid pipeline.

DOCUMENTED PATH — requires ROS2 (Jazzy or Humble) on Ubuntu.
Cannot be tested on Windows. Provided as ready-to-use source code.

Package structure for colcon build:
  ps26053_lidar_mapping/
    src/pipeline/ros2_nodes/
      __init__.py
      segmentation_node.py  (this file)
      grid_node.py

To build:
  cd <ros2_ws>/src
  ln -s /path/to/ps26053_lidar_mapping .
  cd ..
  colcon build --packages-select ps26053_lidar_mapping

To run:
  source install/setup.bash
  ros2 run ps26053_lidar_mapping segmentation_node
  ros2 run ps26053_lidar_mapping grid_node
"""

# This file documents the ROS2 node architecture.
# It requires rclpy, sensor_msgs, std_msgs which are only available in ROS2.

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Int32MultiArray
    import numpy as np
    import struct
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


if ROS2_AVAILABLE:
    class SegmentationNode(Node):
        """ROS2 node that receives PointCloud2, runs segmentation, publishes labels.

        Subscribes to: /velodyne_points (PointCloud2)
        Publishes to: /segmentation/labels (Int32MultiArray)
                      /segmentation/points_classified (PointCloud2 with class field)
        """

        def __init__(self):
            super().__init__("segmentation_node")

            # Import segmentation backend
            from src.segmentation.pointnet_infer import PointNetBackend
            self._backend = PointNetBackend()

            self._sub = self.create_subscription(
                PointCloud2, "/velodyne_points", self._on_pointcloud, 10
            )
            self._pub_labels = self.create_publisher(
                Int32MultiArray, "/segmentation/labels", 10
            )
            self.get_logger().info("SegmentationNode started")

        def _on_pointcloud(self, msg: PointCloud2):
            """Process incoming point cloud."""
            points = self._pointcloud2_to_numpy(msg)
            labels = self._backend.predict(points)

            label_msg = Int32MultiArray()
            label_msg.data = labels.tolist()
            self._pub_labels.publish(label_msg)

        @staticmethod
        def _pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
            """Convert PointCloud2 to (N, 4) numpy array."""
            n_points = msg.width * msg.height
            data = np.frombuffer(msg.data, dtype=np.float32)
            point_step = msg.point_step // 4  # float32 stride
            points = data.reshape(n_points, point_step)[:, :4]
            return points.copy()


    class GridNode(Node):
        """ROS2 node that receives classified points and maintains the adaptive grid.

        Subscribes to: /velodyne_points (PointCloud2)
                       /segmentation/labels (Int32MultiArray)
        Publishes to: /adaptive_grid/markers (via GridVizNode)
        """

        def __init__(self):
            super().__init__("grid_node")

            from src.grid.adaptive_grid import AdaptiveGrid
            from src.grid.resolution import load_grid_config, GridConfig
            from src.viz.rviz_markers import GridVizNode

            config = load_grid_config()
            self._grid = AdaptiveGrid(config)
            self._frame_id = 0

            self._points_buffer = None
            self._labels_buffer = None

            self._sub_points = self.create_subscription(
                PointCloud2, "/velodyne_points", self._on_points, 10
            )
            self._sub_labels = self.create_subscription(
                Int32MultiArray, "/segmentation/labels", self._on_labels, 10
            )

            # Visualization
            from visualization_msgs.msg import MarkerArray
            self._pub_markers = self.create_publisher(
                MarkerArray, "/adaptive_grid/markers", 10
            )

            self._timer = self.create_timer(0.1, self._tick)
            self.get_logger().info("GridNode started")

        def _on_points(self, msg):
            self._points_buffer = SegmentationNode._pointcloud2_to_numpy(msg)

        def _on_labels(self, msg):
            self._labels_buffer = np.array(msg.data, dtype=np.int32)

        def _tick(self):
            if self._points_buffer is not None and self._labels_buffer is not None:
                self._grid.insert(
                    self._points_buffer[:, :3],
                    self._labels_buffer,
                    self._frame_id,
                )
                self._grid.decay_dynamic_cells(self._frame_id)
                self._frame_id += 1

                # Publish visualization
                from src.viz.rviz_markers import create_marker_array
                marker_msg = create_marker_array(self._grid)
                self._pub_markers.publish(marker_msg)

                self._points_buffer = None
                self._labels_buffer = None
