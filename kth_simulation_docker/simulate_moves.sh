#!/bin/bash
trap "exit" INT
cd /tmp
start_time="(date -u +%s)"
while [ true ]; do
  roslaunch kth_navigation simulation.launch
  killall -q -9 roscore
  killall -q -9 rosmaster
  killall -q -9 rosout
  killall -q -9 gzserver
  echo "$(ls)"
  scp -r -o StrictHostKeyChecking=no -i "$HOME/.ssh/bombadil_key.pem" "$KTH_WORLD"* "whitesea@bombadil.engr.oregonstate.edu:workspace"
  cur_time="$(date -u +%s)"
  elapsed="$(($cur_time-$start_time))"
  if [ $elapsed -gt 18000 ]; then
     echo "Done simulating";
     exit
  fi
done
