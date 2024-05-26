import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf.transformations as tft
from geometry_msgs.msg import Point
from squaternion import Quaternion

from path_generator import PathGenerator


TIME_DELTA = 0.1


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, action_space):
        self.odom_x = 0
        self.odom_y = 0

        self.start = [0.0, 0.0]
        self.end = [0.0, 0.0]
        self.floorplan = None

        self.goal_trajectory = [(0.0, 0.0)]
        self.action_space = action_space

        self.trajectory_thickness = 0.1
        self.last_odom = None
        self.goal_point_radius = 0.05

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"

        self.set_self_state.pose.position.x = 0.0 
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.018

        quaternion = tft.quaternion_from_euler(0.0, 0.0, -3.12)
        self.set_self_state.pose.orientation.x = quaternion[0]
        self.set_self_state.pose.orientation.y = quaternion[1]
        self.set_self_state.pose.orientation.z = quaternion[2]
        self.set_self_state.pose.orientation.w = quaternion[3]

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath],) 
                        #  stderr=subprocess.DEVNULL)
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("/tracjectory", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        td = TIME_DELTA + 7 if self.last_odom == None else TIME_DELTA
        time.sleep(td)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # trajectory data
        i_state = self.floorplan[:]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        done, is_oof_traj = self.is_out_of_trajectory()

        # Detect if the goal has been reached and give a large positive reward
        if self.goal_reached():
            target = True
            done = True

        goal_x = self.odom_x - self.goal_trajectory[-1][0]
        goal_y = self.odom_y - self.goal_trajectory[-1][1]
        robot_state = np.array([self.odom_x, self.odom_y, angle, goal_x, goal_y])
        robot_state = (robot_state - robot_state.mean()) / robot_state.std()
        state = [i_state, robot_state]
        reward = self.get_reward(target, is_oof_traj, angle)
        # if target or done:
            # print("reward ", reward, ",done ", done, ", target ", target)
        return state, reward, done, target

    def reset(self, episode):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        
        # max_dist = 4.0 + 0.01 * episode 
        pg = PathGenerator(4.0, 6.0, [-25, 25], [0, 22], 2)
        self.goal_trajectory = pg.generate_path()
        self.start = pg.start_point
        self.end = pg.end_point
        angle = pg.get_orientation(self.goal_trajectory[0], self.goal_trajectory[1])
        
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state
        
        object_state.pose.position.x = self.start[0]
        object_state.pose.position.y = self.start[1]
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y
        
        self.publish_markers([0.0, 0.0])
        self.draw_trajectory()

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        quaternion = Quaternion(
            object_state.pose.orientation.w,
            object_state.pose.orientation.x,
            object_state.pose.orientation.y,
            object_state.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        robot_state = [self.odom_x, self.odom_y, angle] + [0.0] * 2

        self.floorplan = pg.generate_floor_plan(self.goal_trajectory)
        
        state = [self.floorplan[:], robot_state]
        return state

    def sample_action(self):
        return [random.choice(self.action_space[0]), random.choice(self.action_space[1])]
    
    def get_action(self, values):
        scores = values.squeeze(0).reshape(-1, 2, 5).max(dim=-1)[1].squeeze(0)
        return [self.action_space[0][scores[0]], self.action_space[1][scores[1]]]

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = abs(action[0])
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 3
        marker.pose.position.y = -22

        markerArray.markers.append(marker)
        self.publisher2.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[1])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.b = 1.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 3
        marker2.pose.position.y = -22.2

        markerArray2.markers.append(marker2)
        self.publisher3.publish(markerArray2)
    
    def draw_trajectory(self):
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

        for point in self.goal_trajectory:
            p = Point()
            p.x, p.y = point[0], point[1]
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)

        self.publisher.publish(marker_array)

    def is_out_of_trajectory(self):
        x, y = self.odom_x, self.odom_y
        gt = self.goal_trajectory
        for i in range(len(gt) - 1):
            x2, y2 = gt[i]
            x3, y3 = gt[i + 1]

            dist = round(self.point_to_line_distance(x, y, x2, y2, x3, y3), 1)

            if dist <= self.trajectory_thickness:
                return False, False
        # print("Robot is out of trajectory. X:", x, "Y:", y )  
        return True, True

    def goal_reached(self):
        x, y = self.odom_x, self.odom_y
        end_x, end_y = self.goal_trajectory[-1]
        distance = math.sqrt((end_x - x)**2 + (end_y - y)**2)
        return distance < self.goal_point_radius

    def get_reward(self, target, is_oof_traj, angle):
        if target:
            return 100.0
        elif is_oof_traj:
            return -100.0
        else:
            distance = self.get_distance([self.odom_x, self.odom_y], self.goal_trajectory[-1])
            max_distance = self.get_distance(self.goal_trajectory[0], self.goal_trajectory[-1])
            normalized_distance = 1 - (distance / max_distance)
            scaled_distance = normalized_distance * 5.0
            return scaled_distance
        
    @staticmethod
    def get_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    @staticmethod
    def point_to_line_distance(x, y, x1, y1, x2, y2):
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return numerator / denominator