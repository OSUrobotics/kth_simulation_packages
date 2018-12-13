# You need to build this from the context of the parent directory
FROM ros:kinetic-ros-core-xenial
MAINTAINER Austin Whitesell (whitesea@oregonstate.edu)
SHELL ["/bin/bash", "-c"]
RUN rm -rf /var/lib/apt/lists/*
RUN apt clean
RUN apt update
RUN apt-get update
RUN apt-get install wget
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list
RUN wget http://packages.osrfoundation.org/gazebo.key -O - |  apt-key add -
RUN apt-get update
RUN apt-get install -y ros-kinetic-gazebo9-*
RUN apt-get install -y ros-kinetic-catkin
RUN apt-get install -y ros-kinetic-opencv-candidate
RUN apt-get install -y ros-kinetic-costmap-2d
RUN apt-get install -y ros-kinetic-robot-controllers*
RUN apt-get install -y ros-kinetic-map-server
RUN apt-get install -y ros-kinetic-xacro
RUN apt-get install -y ros-kinetic-rgbd-launch
RUN apt-get install -y ros-kinetic-amcl
RUN apt-get install -y ros-kinetic-move-base
RUN apt-get install -y ros-kinetic-robot-state-publisher
RUN apt-get install -y ssh
RUN apt-get install -y psmisc
RUN apt-get remove -y wget
RUN useradd -ms /bin/bash rosuser

USER rosuser
WORKDIR /home/rosuser/
RUN mkdir -p catkin_ws/src
COPY --chown=rosuser:rosuser ros_packages/fetch_depth_layer /home/rosuser/catkin_ws/src/fetch_depth_layer
COPY --chown=rosuser:rosuser ros_packages/fetch_description /home/rosuser/catkin_ws/src/fetch_description
COPY --chown=rosuser:rosuser ros_packages/fetch_gazebo /home/rosuser/catkin_ws/src/fetch_gazebo
COPY --chown=rosuser:rosuser ros_packages/fetch_maps /home/rosuser/catkin_ws/src/fetch_maps
COPY --chown=rosuser:rosuser ros_packages/fetch_navigation /home/rosuser/catkin_ws/src/fetch_navigation
COPY --chown=rosuser:rosuser ros_packages/kth_navigation /home/rosuser/catkin_ws/src/kth_navigation
COPY --chown=rosuser:rosuser ros_packages/move_base /home/rosuser/catkin_ws/src/move_base
COPY --chown=rosuser:rosuser bombadil_key.pem /home/rosuser/.ssh/
COPY --chown=rosuser:rosuser kth_simulation_docker/kth_entrypoint.sh /home/rosuser/
COPY --chown=rosuser:rosuser kth_simulation_docker/simulate_moves.sh /home/rosuser/
# COPY --chown=rosuser:rosuser kth_simulation_key.pem /home/rosuser/.ssh/
RUN cd catkin_ws/src && source /opt/ros/kinetic/setup.bash && catkin_init_workspace
RUN cd catkin_ws && source /opt/ros/kinetic/setup.bash && catkin_make
RUN chmod u+x kth_entrypoint.sh
ENTRYPOINT ["/home/rosuser/kth_entrypoint.sh"]
