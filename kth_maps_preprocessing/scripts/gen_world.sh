#!/bin/bash -e


for d in /home/$USER/catkin_ws/src/kth_navigation/models/*/model.sdf; do
  python buildworld.py $d
done
