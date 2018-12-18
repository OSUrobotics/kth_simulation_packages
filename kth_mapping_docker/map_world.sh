#!/bin/bash
roslaunch kth_navigation map_known.launch
cd /tmp
rosbag filter $KTH_WORLD.bag "$KTH_WORLD"_new.bag 'topic != "/tf" or topic == "/tf" and m.transforms[0].header.frame_id != "map" and m.transforms[0].child_frame_id != "map"'
roscore &
roslaunch --wait kth_navigation gmapping.launch &
roslaunch --wait kth_navigation play_bag.launch
mkdir $KTH_WORLD
cd $KTH_WORLD
rosrun map_server map_saver
killall -q -9 roscore
killall -q -9 rosmaster
killall -q -9 rosout
killall -q -9 gzserver
