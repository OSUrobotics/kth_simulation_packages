#!/bin/bash
source catkin_ws/devel/setup.bash
bash map_world.sh
cd /tmp
scp -r -o StrictHostKeyChecking=no -i "$HOME/.ssh/bombadil_key.pem" "$KTH_WORLD" "whitesea@bombadil.engr.oregonstate.edu:workspace/navigation_analysis_packages/ros_maps/"
exit
