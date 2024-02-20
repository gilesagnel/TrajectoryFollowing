from visualization_msgs.msg import Marker, MarkerArray
import rospy
import numpy as np
from geometry_msgs.msg import Point


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def create_trajectory_marker(points):
    marker_array = MarkerArray()
    
    line_marker = Marker()
    line_marker.header.frame_id = "odom"  
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.1  # Set the line width

    line_marker.color.r = 0.0  # Set the color (red)
    line_marker.color.g = 1.0
    line_marker.color.b = 0.0
    line_marker.color.a = 1.0  # Set the alpha (transparency)

    line_marker.points = []

    for point in points:
        p = Point()
        p.x, p.y = point[0], point[1]
        line_marker.points.append(p)

    marker_array.markers.append(line_marker)

    return marker_array

def create_trajectory(data):
    trajectory = [(0.0, 0.0)]
    for d in data:
        axis, (_, end) = d
        x, y = trajectory[-1]
        start = x if axis == "x" else y
        step = 0.1 if start < end else -0.1
        points = [(round(i, 1), y) if axis == "x" else (x, round(i, 1)) for i in np.arange(start, end, step)]
        trajectory.extend(points)
    return trajectory


if __name__ == "__main__":
    data = [("x" ,(0.0 , 8.5)),   ("y", (0.0 , 9.1)),  ("x", (8.5 , 14.4)), ("y", (9.1, -4.2)), ("x", (14.4 , 26.3)), 
            ("y", (-4.2, 0.2)), ("x", (26.3 , 42.4)),  ("y", (0.2 , 8.0)), ("x", (42.4 , 46.2))]
    trajectory = create_trajectory(data)
    
    rospy.init_node("trajectory_visualizer")
    marker_array = create_trajectory_marker(trajectory)

    marker_publisher = rospy.Publisher("/tracjectory", MarkerArray, queue_size=10)

    rate = rospy.Rate(1)  # Adjust the rate according to your needs

    while not rospy.is_shutdown():
        marker_publisher.publish(marker_array)
        rate.sleep()
