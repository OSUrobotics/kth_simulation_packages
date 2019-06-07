#!/usr/bin/env python
from toolkit import *
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    print('processing ' + world)
    if not os.path.isdir('/home/whitesea/workspace/data_post_processing/data/' +world + '_ros/'):
        print('no data for world ' + world)
        sys.exit(1)
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/accumulator_data_5.npy'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    data_path = '/home/whitesea/workspace/data_post_processing/data/' + world +'_ros/'
    accumulated_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/accumulated.npy'
    label_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/failure_labels.npy'
    if os.path.isfile(label_file):
        pass
    else:
        failure_labels = label_failures(data_path)
        np.save(label_file,failure_labels)

    if os.path.isfile(accumulated_file):
        pass
    else:
        accumulated = generate_accumulated_data(output_folder,window_size=5,stride=.25)
        np.save(accumulated_file,accumulated)
        accumulated = (accumulated / float(accumulated.max())) * 255.0
        accumulated_image = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/accumulated.pgm'
        imsave(accumulated_image,accumulated)

    data, im = generate_accumulated_training_data(output_folder,samples=500,window_size=5.0,threshold = 0.9999,draw = True)
    np.save(training_data_file,data)
    imsave(output_folder + 'training_windows.png',im)
