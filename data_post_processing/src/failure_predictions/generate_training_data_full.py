#!/usr/bin/env python

from toolkit import *
import sys



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    if not os.path.isdir('/home/whitesea/workspace/data_post_processing/data/' + world + '_ros/'):
        print('no data for world ' + world)
        sys.exit(1)
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/failure_training_data_4_5.npy'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    data_path = '/home/whitesea/workspace/data_post_processing/data/' + world +'_ros/'
    heatmap_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/blurred_fails.npy"
    if os.path.isfile(heatmap_file):
        pass
    else:
        heatmap = make_heatmap(data_path)
        np.save(heatmap_file, heatmap)



    label_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/labels.npy"
    if os.path.isfile(label_file):
        pass
    else:
        failed_labels = generate_failure_labels(heatmap_file)
        visited_labels = generate_visited_labels(data_path)
        labels = np.maximum(visited_labels,failed_labels)
        np.save(label_file, labels)
        labels = labels * 127
        label_image = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/labels.pgm"
        imsave(label_image,labels)

    data = generate_training_data(output_folder, samples=500, window_size=4.5,draw = False)
    np.save(training_data_file, data)
