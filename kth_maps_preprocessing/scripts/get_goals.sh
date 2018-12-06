#!/bin/bash

RELATIVE_PATH="`dirname \"$0\"`"
trap "exit" INT
for d in /home/$USER/workspace/kth_maps/maps/*/; do
  python findGoals.py $d
done
