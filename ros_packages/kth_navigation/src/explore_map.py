#!/usr/bin/env python
import rospy, roslib, cv2, message_filters, time, tf, yaml
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float64
import numpy as np
from tf.transformations import euler_from_quaternion
from math import atan2, pi, sqrt
from nav_msgs.srv import GetMap
from kth_navigation.srv import GetPose
from time import sleep
from copy import deepcopy

initialpose = PoseWithCovarianceStamped()
initialpose.pose.pose.position.x = 0.0
initialpose.pose.pose.position.y = 0.0
initialpose.pose.pose.position.z = 0.0

initialpose.pose.pose.orientation.x = 0.0
initialpose.pose.pose.orientation.y = 0.0
initialpose.pose.pose.orientation.z = 0.0
initialpose.pose.pose.orientation.w = 1.0

initialpose.pose.covariance = [0.0] * 36

class slam_explorer():
    def __init__(self):
        # Set up move_base and map subscriber
        self.sub1 = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.callback1)
        self.pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.pub1 = rospy.Publisher('/initialpose',PoseWithCovarianceStamped,queue_size=10)
        self.pub2 = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        try:
            goals_file = rospy.get_param('explorer/goals_file')
        except KeyError:
            rospy.logerr("explorer has crashed: parameters are not set")

        generator = yaml.load_all(open(goals_file, 'r'))
        self.goals = None
        for thing in generator:
            self.goals = thing
        goal = self.goals.pop()
        rospy.wait_for_service('/move_base/make_plan')
        rospy.wait_for_service('/amcl_pose_server')
        rospy.sleep(10)
        self.get_amcl_pose = rospy.ServiceProxy('/amcl_pose_server', GetPose)
        initialpose.header.stamp = rospy.Time.now()
        initialpose.header.frame_id = "map"
        self.pub1.publish(initialpose)
        self.pub.publish(self.goal_to_pose(goal))
        rospy.loginfo('Explorer started')

    def callback1(self, msg):
        done_status = [3,4,5]
        rospy.loginfo(str(len(self.goals)) + " locations left")
        if msg.status.status in done_status:
            goal = None
            goal = Twist()
            goal.linear.x = 0.0
            goal.linear.y = 0.0
            goal.linear.z = 0.0
            goal.angular.x = 0.0
            goal.angular.y = 0.0
            goal.angular.z = 1.0

            angle = 0.0
            cur_pose = self.get_amcl_pose().msg.pose.pose
            quat = [cur_pose.orientation.x,cur_pose.orientation.y,cur_pose.orientation.z,cur_pose.orientation.w]
            roll,pitch,yaw0 = euler_from_quaternion(quat)

            while angle < 2*pi:
                self.pub2.publish(goal)
                cur_pose = self.get_amcl_pose().msg.pose.pose
                quat = [cur_pose.orientation.x,cur_pose.orientation.y,cur_pose.orientation.z,cur_pose.orientation.w]
                roll,pitch,yaw1 = euler_from_quaternion(quat)
                addition = yaw1 - yaw0
                if addition < 0:
                    addition = addition + 2*pi

                angle = angle + addition
                yaw0 = yaw1

            goal.angular.z = 0.0
            self.pub2.publish(goal)
            try:
                goal = self.goals.pop()
            except:
                goal = None
                pass
            if goal:
                self.pub.publish(self.goal_to_pose(goal))

            else:
                rospy.signal_shutdown("Done exploring")

    def goal_to_pose(self,goal):
        pose = PoseStamped()
        pose.pose.position.x = goal['position']['x']
        pose.pose.position.y = goal['position']['y']
        pose.pose.position.z = goal['position']['z']
        pose.pose.orientation.x = goal['orientation']['x']
        pose.pose.orientation.y = goal['orientation']['y']
        pose.pose.orientation.z = goal['orientation']['z']
        pose.pose.orientation.w = goal['orientation']['w']

        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        return pose

if __name__ == "__main__":
    rospy.init_node('slam_explorer')
    explorer = slam_explorer()

    rospy.spin()
