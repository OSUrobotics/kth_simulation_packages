#!/bin/bash
trap "exit" INT
cd /tmp
start_time="$(date -u +%s)"
scp -r -o StrictHostKeyChecking=no -i "$HOME/.ssh/kth_simulation_key.pem" "ubuntu@$STORAGE_DOMAIN:kth_ros_maps/$KTH_WORLD" "$HOME/catkin_ws/src/kth_navigation/ros_maps/"
while [ true ]; do
  roslaunch kth_navigation simulation.launch
  killall -q -9 roscore
  killall -q -9 rosmaster
  killall -q -9 rosout
  killall -q -9 gzserver
  scp -r -o StrictHostKeyChecking=no -i "$HOME/.ssh/kth_simulation_key.pem" "$KTH_WORLD"* "ubuntu@$STORAGE_DOMAIN:data_post_processing/input/"
  cur_time="$(date -u +%s)"
  elapsed="$(($cur_time-$start_time))"
  if [ $elapsed -gt 259200 ]; then
     echo "Done simulating";
     exit
  fi
done
