The KTH Simulation Package
==========================

This repository contains all the code used for generating simulated navigation
data in a subset of the environments from the KTH floorplan dataset

`Aydemir, Alper, Patric Jensfelt, and John Folkesson. "What can we learn from 38,000 rooms? Reasoning about unexplored space in indoor environments." 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012.`

There are 3 main components to using this software, which is provided free as is.

##Preprocessing

The kth_maps_preprocessing package contains code for taking the xml files included in the original floorplan dataset and converting them into
Gazebo world files to be used for simulation, as well as map files for use with ROS

##Simulation

The kth_simulation_docker folder contains instructions for creating a docker image to simulate
a robot navigating in these environments, and describes how the data is managed.

The ros_packages folder contains a modified move_base that publishes when recovery behaviors are executed,
as well as a custom package (kth_navigation) for generating navigation goals and saving the data. It also contains
the source files for simulating a Fetch robot in Gazebo, cloned from their repository:
`https://github.com/fetchrobotics`

##Postprocessing

The meat of the project lives in the data_post_processing folder. This is where our attempts to create a neural network
to predict navigation behaviors and failures live. All of the code in this work is of research-quality, so use at your own risk. I am happy to assist
wherever I can, leave me a message or report an issue. Hopefully I will have some time to continue to clean this code up and improve documentation.
