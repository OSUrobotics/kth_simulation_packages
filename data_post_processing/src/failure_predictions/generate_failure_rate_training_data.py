#!/usr/bin/env python

from toolkit import *
import sys



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    print(bcolors.HEADER +  'PROCESSING WORLD: ' + world + bcolors.ENDC)
    data_path = '/home/whitesea/workspace/data_post_processing/compressed_data/' + world +'/'
    if not os.path.isdir(data_path):
        print(bcolors.FAIL +  'no data for world ' + world + bcolors.ENDC)
        sys.exit(1)
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/failure_rate_training_data.npz'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    heatmap_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/failure_rates.npy"
    if os.path.isfile(heatmap_file):
        pass
    else:
        heatmap = generate_failure_rate_heatmap(data_path,resolution_scaling=128, threshold = 1.0)
        np.save(heatmap_file, heatmap)
        # try:
        # except:
        #     f = open('log.txt','a+')
        #     f.write('problem with %s\n' % world)
        #     sys.exit()
    # data = generate_failure_rate_training_data(output_folder,samples=200,window_size=12.8)
    # np.savez_compressed(training_data_file,data)


    #
    # label_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/labels.npy"
    # if os.path.isfile(label_file):
    #     pass
    # else:
    #     failed_labels = generate_failure_labels(heatmap_file)
    #     visited_labels = generate_visited_labels(data_path)
    #     labels = np.maximum(visited_labels,failed_labels)
    #     np.save(label_file, labels)
    #     labels = labels * 127
    #     label_image = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/labels.pgm"
    #     imsave(label_image,labels)
    #
    # data = generate_training_data(output_folder, samples=500, window_size=4.5,draw = False)
    # np.save(training_data_file, data)
