#!/usr/bin/env python
import rospy, actionlib, os, cv2
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String
# from std_srvs.srv import *
from move_base_msgs.msg import MoveBaseActionResult, MoveBaseGoal, MoveBaseAction
from actionlib_msgs.msg import *
from kth_navigation.srv import *
from nav_msgs.srv import GetMap
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Pose
import numpy as np
from time import time


class Simulator:
    def __init__(self):
        rospy.init_node('simulator')
        try:
            self.robot = rospy.get_param('simulator/robot')
            self.g_frame = rospy.get_param('simulator/gazebo_frame')
            self.a_frame = rospy.get_param('simulator/amcl_frame')
            self.results_folder = rospy.get_param('simulator/results_folder')
            self.radius = rospy.get_param('/move_base/global_costmap/inflater/robot_radius')
        except KeyError:
            # print("keyerror")
            rospy.logerr("simulator has crashed: parameters are not set")

        # Setup some variables we need
        self.start_time = rospy.get_time()
        self.stop_time = rospy.get_time()
        self.recovery_locations = []
        self.stopped = False
        self.move_result = None


        # Get my process id
        self.id = str(os.getpid())

        # Get Service proxies for sending gazebo poses, actions, and getting amcl pose
        # rospy.wait_for_service('gazebo/set_model_state')
        # self.set_gazebo_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_gazebo_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('amcl_pose_server')
        self.get_amcl_pose = rospy.ServiceProxy('/amcl_pose_server', GetPose)
        # rospy.wait_for_service('move_base/clear_costmaps')
        # self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        self.send_goal = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.send_goal.wait_for_server(rospy.Duration(60))

        # Subscribe to the results of moves
        self.move_result_sub = rospy.Subscriber('move_base/result', MoveBaseActionResult, self.callback)
        self.subscriber2 = rospy.Subscriber('recovery_status',String,self.recovery_callback)

        # get the map
        rospy.wait_for_service('/static_map')
        self.getMap = rospy.ServiceProxy('/static_map', GetMap)

        self.locations = self.get_locations()
        np.random.seed(int(time()))

        self.failed_attempts = 0
        # Publish initial pose estimates:



    def simulate(self):
        data = None
        # Let's run some tests
        # rospy.loginfo("got to simulate")
        for t in range(50):
            self.stopped = False
            # Try to set the starting point and localize
            self.start_time = rospy.get_time()
            self.send_goal.send_goal(self.get_goal())
            amcl_path = []
            gazebo_path = []
            while(not self.stopped):
                amcl_path.append(self.get_amcl_pose().msg.pose.pose)
                gazebo_path.append(self.get_gazebo_state(self.robot,self.g_frame).pose)
                rospy.sleep(1)
                cur_time = rospy.get_time()
                if (cur_time - self.start_time) > 1800:
                    # robot is taking too long, possibly stuck
                    rospy.signal_shutdown(0)
            data = dict()
            data['result'] = self.move_result
            data['move_time'] = self.stop_time - self.start_time
            data['amcl_path'] = amcl_path
            data['gazebo_path'] = gazebo_path
            data['recoveries'] = self.recovery_locations
            self.write_data(data)
            self.recovery_locations = []

            if self.failed_attempts > 4:
                rospy.signal_shutdown(0)
        rospy.signal_shutdown(0)


    def get_locations(self):
        resp = self.getMap()
        grid = resp.map
        map_data = grid.data
        res = grid.info.resolution
        width = grid.info.width
        height = grid.info.height
        origin = grid.info.origin.position

        open_dict = dict()
        closed_dict = dict()
        # convert to numpy array for inflation
        raw_map = np.zeros((height,width))
        inflated_map = np.zeros((height,width))
        for i in range(height - 1):
            for j in range(width - 1):
                if map_data[i*width + j] == 0:
                    raw_map[i,j] = 0
                    inflated_map[i,j] = 0
                else:
                    raw_map[i,j] = 1
                    inflated_map[i,j] = 1

        # inflate the map
        for i in range(height):
            for j in range(width):
                open_dict[(j,i)] = False
                closed_dict[(j,i)] = False
                if raw_map[i,j]:
                    inflated_map = cv2.circle(inflated_map,(j,i), int(self.radius / res), color=1,thickness=-1)

        x_org = int(-origin.x / res)
        y_org = int(-origin.y / res)
        open_dict[(x_org,y_org)] = True
        open_list = [(x_org,y_org)]
        locations = []
        while open_list:
            sample = open_list.pop(0)
            x = sample[0]
            y = sample[1]
            closed_dict[(x,y)] = True
            locations.append(sample)
            for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                if i < 0 or j < 0 or i >= width or j >= height or open_dict[(i,j)]:
                    pass
                else:
                    if inflated_map[j,i] == 1:
                        open_dict[(i,j)] = True
                    else:
                        open_dict[(i,j)] = True
                        open_list.append((i,j))

        # convert locations to metric coordinates
        for i in range(len(locations)):
            locations[i] = [(locations[i][0] * res + origin.x, locations[i][1] * res + origin.y)]

        rospy.loginfo(len(locations))
        return locations

    def get_goal(self):
        goal = MoveBaseGoal()
        goal_pose = Pose()
        x,y = self.locations[np.random.choice(range(len(self.locations)))][0]
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        goal.target_pose.header.frame_id = self.a_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        return goal

    def callback(self,msg):
        self.stop_time = rospy.get_time()
        self.move_result = int(msg.status.status)
        self.stopped = True
        if self.move_result == 3:
            self.failed_attempts = 0
        else:
            self.failed_attempts = self.failed_attempts + 1

    def recovery_callback(self,msg):
        entry = dict()
        entry["amcl_pose"] = self.get_amcl_pose().msg.pose.pose
        entry["gazebo_pose"] = self.get_gazebo_state(self.robot,self.g_frame).pose
        entry["recovery_behavior"] = msg.data
        self.recovery_locations.append(entry)

    def write_data(self,data):
        filename = self.results_folder + '/data.npy'
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        if os.path.isfile(filename):
            # Results already exists
            results = np.load(filename)
            results = results.item()
            results['data_points'] = results['data_points'] + [data]
            np.save(filename,results)

        else:
            results = dict()
            results['data_points'] = [data]
            np.save(filename,results)

if __name__ == "__main__":
    simulator = Simulator()
    rospy.sleep(10)
    simulator.simulate()
    rospy.spin()
