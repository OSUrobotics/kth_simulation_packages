#!/usr/bin/env python
import rospy, roslib, cv2, message_filters, time, tf, yaml
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped
from std_msgs.msg import Float64
import numpy as np
from tf.transformations import quaternion_from_euler
from math import atan2, pi, sqrt
from nav_msgs.srv import GetMap
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
        rospy.sleep(10)
        initialpose.header.stamp = rospy.Time.now()
        initialpose.header.frame_id = "map"
        self.pub1.publish(initialpose)
        self.turned = False
        self.pub.publish(self.goal_to_pose(goal))
        rospy.loginfo('Explorer started')

    def callback1(self, msg):
        done_status = [3,4,5]
        rospy.loginfo('Callback')
        if msg.status.status in done_status:
            goal = None
            turn = quaternion_from_euler(0,0,360)
            if not self.turned:
                goal = dict(
                    position = dict(
                        x = 0.0,
                        y = 0.0,
                        z = 0.0,
                    ),
                    orientation = dict(
                        x = turn[0],
                        y = turn[1],
                        z = turn[2],
                        w = turn[3],
                    )
                )
                self.turned = True
                goal = self.goal_to_pose(goal)
                goal.header.frame_id = "base_link"
                self.pub.publish(goal)
            else:
                try:
                    goal = self.goals.pop()
                except:
                    pass
                if goal:
                    self.turned = False
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
