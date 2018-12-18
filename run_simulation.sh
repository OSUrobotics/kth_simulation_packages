#!/bin/bash
trap "exit" INT
MODES=(true
false)

for i in "${MODES[@]}"
do
   docker run -e "KTH_WORLD=0510025536_Layout1_PLAN4" -e "USE_ROS_MAP=$i" -d  kth_simulation
done
sleep 2
start_time="$(date -u +%s)"

while true
do
   var=$(docker ps | grep kth_simulation | wc -l)
   if [ $var -eq 0 ]; then
      echo "Done simulating";
      exit
   fi
   cur_time="$(date -u +%s)"
   elapsed="$(($cur_time-$start_time))"
   if [ $elapsed -gt 18000 ]; then
      echo "Done simulating";
      exit
   fi
done
