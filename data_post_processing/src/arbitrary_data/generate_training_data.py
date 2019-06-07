#!/usr/bin.env python

from toolkit import *
from generate_data import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/relative_training_data.npy'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    relative_values_file = output_folder + "/relative_values.npy"
    if os.path.isfile(relative_values_file):
        pass
    else:
        im = imread(output_folder + 'map.pgm')
        relative_values = generate_relative_values(im)
        np.save(relative_values_file,relative_values)

    data = generate_relative_training_data(output_folder,samples=100,window_size=10)
    np.save(training_data_file,data)
