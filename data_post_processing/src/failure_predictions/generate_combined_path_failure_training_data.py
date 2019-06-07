#!/usr/bin/env python

from toolkit import *
import sys
from PIL import Image



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    data_path = '/home/whitesea/workspace/data_post_processing/compressed_data/' + world +'/'
    if not os.path.isdir(data_path):
        print('no data for world ' + world)
        sys.exit(1)
    im = imread(data_path + 'map.pgm')
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/combined_path_failure_training_data_v2.npz'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    heatmap_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/interpolated_blurred_failure_rates.npy"
    visits_file = output_folder + 'visits.npy'
    if os.path.isfile(heatmap_file):
        heatmap = np.load(heatmap_file)
        pass
    else:
        heatmap = generate_failure_rate_heatmap(data_path,resolution_scaling=16.0)
        np.save(heatmap_file, heatmap)
    if os.path.isfile(visits_file):
        visits = np.load(visits_file)
        pass
    else:
        visits = get_visits(data_path)
        np.save(visits_file, visits)

    im = Image.fromarray(im)
    im = im.resize((1024,1024),Image.ANTIALIAS)
    im = np.array(im)
    im = im.reshape(1024,1024,1)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((1024,1024),Image.ANTIALIAS)
    heatmap = np.array(heatmap)
    heatmap = heatmap.reshape((heatmap.shape[0],heatmap.shape[1],1))
    visits = visits / visits.max()
    visits = visits * 255.0
    visits = visits.astype(np.uint8)
    visits = Image.fromarray(visits)
    visits = visits.resize((1024,1024),Image.ANTIALIAS)
    visits = np.array(visits)
    visits = visits.reshape((visits.shape[0],visits.shape[1],1))
    out = np.concatenate([im,visits,heatmap], axis=-1)
    out = out.astype(np.uint8)

    data = rotate_data(out)

    np.savez_compressed(training_data_file,data)
