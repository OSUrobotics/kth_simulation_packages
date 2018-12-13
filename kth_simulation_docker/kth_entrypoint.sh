#!/bin/bash
source catkin_ws/devel/setup.bash
bash simulate_moves.sh
# bash map_world.sh
# cd /tmp
# scp -r -o StrictHostKeyChecking=no -i "$HOME/.ssh/kth_simulation_key.pem" "$KTH_WORLD" "ubuntu@$BLOCK_STORAGE_DOMAIN:data"
# exit
