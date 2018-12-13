#!/usr/bin/env python
import rospy, roslib, cv2, message_filters, time, tf
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

class slam_explorer():
    def __init__(self):
        self.current_pose = Pose()
        self.listener = tf.TransformListener()

        # Set up move_base and map subscriber
        self.sub1 = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.callback1)
        self.pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        rospy.wait_for_service('/dynamic_map')
        rospy.sleep(2)
        self.getMap = rospy.ServiceProxy('/dynamic_map', GetMap)
        self.failures = 0
        sleep(2)
        self.pub.publish(self.get_goal())

    def callback1(self, msg):
        done_status = [3,4,5]
        if msg.status.status in done_status:
            if msg.status.status in done_status[1:]:
                self.failures = self.failures + 1
                if self.failures > 5:
                    rospy.signal_shutdown("Done exploring")
            else:
                self.failures = 0

            self.pub.publish(self.get_goal())

    # Define helper functions
    def euclidian_dist(self,point1,point2):
        dx = point2.position.x - point1.position.x
        dy = point2.position.y - point1.position.y
        return sqrt(dx*dx + dy*dy)

    def get_goal(self):
        resp = self.getMap()
        grid = resp.map
        map_data = grid.data
        resolution = grid.info.resolution
        width = grid.info.width
        height = grid.info.height
        origin = grid.info.origin
        locations = []
        for i in range(width - 1):
            for j in range(height - 1):
                if map_data[j*width + i] == -1:
                    if map_data[j*width + i + 1] == 0:
                        locations.append((i,j))
                    elif map_data[j*width + i - 1] == 0:
                        locations.append((i,j))
                    elif map_data[(j+1)*width + i] == 0:
                        locations.append((i,j))
                    elif map_data[(j-1)*width + i] == 0:
                        locations.append((i,j))
        if len(locations) == 0:
            rospy.signal_shutdown("Done exploring")
        ind = np.random.choice(range(len(locations)))
        new_goal = locations[ind]
        newx = new_goal[0] * resolution + origin.position.x
        newy = new_goal[1] * resolution + origin.position.y

        position = Pose()
        position.position.x = newx
        position.position.y = newy
        position.position.z = 0.0
        quat = quaternion_from_euler(0,0,np.random.uniform(-pi,pi))
        position.orientation.x = quat[0]
        position.orientation.y = quat[1]
        position.orientation.z = quat[2]
        position.orientation.w = quat[3]

        goal = PoseStamped()
        goal.pose = position
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        return goal

if __name__ == "__main__":
    rospy.init_node('slam_explorer')
    explorer = slam_explorer()

    rospy.spin()
