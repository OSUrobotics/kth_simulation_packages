Creating the Simulation Docker Image
====================================

This directory contains the dockerfile for building the simulation docker image

##Building the image
To build the image, a change in needed:

If you choose to use the distributed set up with a remote server for data
and map file storage, then you will need to replace the kth_simulation_key.pem
reference to a key that you can use for connecting to your server.

To build the image, navigate to this directory and run the following command:
   `docker build -t kth_simulation -f kth_simulation_docker/Dockerfile .`


##Running the image
The docker image as stands looks for two environment variables to be set at runtime

   1. The domain name of the storage server
   2. The name of the KTH environment to be simulated

The storage server needs to be set up with the following characteristics:
   * A copy of the data_post_processing directory from this package
   * The consolidate_files.sh script running in the background

An example of running simulation image can be called like this:

  `docker run -e "KTH_WORLD=50010539" -e "STORAGE_DOMAIN=<Storage Domain>" -d kth_simulation`

This will run the docker image silently and will send the generated data to the storage server,

The script running on the storage server will automatically move any newly generated data to
`data/<KTH world>/<time stamp>_data.npy`

The docker image runs through the simulate_moves.sh script, which sets the image to
automatically shut down after about 3 days of simulation. This can be adjusted by
changing the value in the if statement just before the end of the script.
