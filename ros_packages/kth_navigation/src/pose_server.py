#! /usr/bin/env python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from kth_navigation.srv import *
mesg = PoseWithCovarianceStamped()

def callback(msg):
    global mesg
    mesg = msg

def handle_service(req):
    global mesg
    return mesg

if __name__ == "__main__":
    rospy.init_node('amcl_pose_server')
    sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, callback)
    s = rospy.Service('amcl_pose_server', GetPose, handle_service)
    rospy.spin()
