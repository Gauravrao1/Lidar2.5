"""
ROS2 Visualization — RViz2 MarkerArray publisher for the adaptive grid.

DOCUMENTED PATH — requires ROS2 (Jazzy or Humble) on Ubuntu.
Cannot be tested on Windows. Provided as ready-to-use source code.

Usage (on a ROS2-enabled system):
  1. colcon build --packages-select ps26053_lidar_mapping
  2. source install/setup.bash
  3. ros2 run ps26053_lidar_mapping grid_viz_node
  4. Open RViz2, add MarkerArray display on topic /adaptive_grid/markers
"""

# rclpy import will fail on non-ROS2 systems — this is expected
try:
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Point, Vector3
    from builtin_interfaces.msg import Duration
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..grid.adaptive_grid import AdaptiveGrid

# Class colors (RGBA, 0-1 range)
CLASS_COLORS = {
    0: (0.0, 0.8, 0.0, 0.7),   # Terrain — green
    1: (0.0, 0.4, 1.0, 0.7),   # Static — blue
    2: (1.0, 0.2, 0.2, 0.8),   # Dynamic — red
    3: (0.5, 0.5, 0.5, 0.3),   # Ignore — gray (rarely shown)
}


def create_marker_array(grid: "AdaptiveGrid", frame_id: str = "velodyne") -> "MarkerArray":
    """Convert AdaptiveGrid cells to an RViz2 MarkerArray.

    Each cell becomes a CUBE marker with:
    - Position at cell center (x, y, z)
    - Scale matching the cell's actual computed size (variable resolution visible)
    - Color based on class (green=T, blue=S, red=D)

    Args:
        grid: The adaptive grid to visualize
        frame_id: TF frame for the markers

    Returns:
        MarkerArray message ready to publish
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS2 is not available in this environment")

    marker_array = MarkerArray()
    cells = grid.get_cells()

    for idx, cell_data in enumerate(cells):
        key, class_label, center_xyz, cell_size_m, last_frame = cell_data

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = frame_id
        marker.ns = "adaptive_grid"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = float(center_xyz[0])
        marker.pose.position.y = float(center_xyz[1])
        marker.pose.position.z = float(center_xyz[2])
        marker.pose.orientation.w = 1.0

        # Scale — matches actual cell size (this is the key visual feature)
        marker.scale = Vector3(
            x=float(cell_size_m),
            y=float(cell_size_m),
            z=float(0.25),  # z_bin_size
        )

        # Color by class
        r, g, b, a = CLASS_COLORS.get(class_label, (0.5, 0.5, 0.5, 0.3))
        marker.color = ColorRGBA(r=r, g=g, b=b, a=a)

        # Lifetime — markers expire after 0.5s if not refreshed
        marker.lifetime = Duration(sec=0, nanosec=500_000_000)

        marker_array.markers.append(marker)

    return marker_array


if ROS2_AVAILABLE:
    class GridVizNode(Node):
        """ROS2 node that publishes the adaptive grid as MarkerArray to RViz2.

        Subscribes to: (none — call update_grid() from the pipeline)
        Publishes to: /adaptive_grid/markers (MarkerArray)

        Usage:
            node = GridVizNode()
            # In your pipeline loop:
            node.update_grid(grid)
            rclpy.spin_once(node)
        """

        def __init__(self, node_name: str = "grid_viz_node"):
            super().__init__(node_name)
            self._pub = self.create_publisher(MarkerArray, "/adaptive_grid/markers", 10)
            self.get_logger().info("GridVizNode started — publishing to /adaptive_grid/markers")

        def update_grid(self, grid: "AdaptiveGrid") -> None:
            """Publish current grid state as MarkerArray."""
            msg = create_marker_array(grid, frame_id="velodyne")
            self._pub.publish(msg)

        def publish_delete_all(self) -> None:
            """Publish a single DELETE_ALL marker to clear RViz."""
            marker_array = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
            self._pub.publish(marker_array)
