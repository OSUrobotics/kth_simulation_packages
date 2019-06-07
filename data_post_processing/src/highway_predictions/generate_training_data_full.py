#!/usr/bin/env python

import sys, os
from PIL import Image

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Not enough arguments passed')
        sys.exit(1)
    world = sys.argv[1]
    print(world)
    if not os.path.isdir('/home/whitesea/workspace/data_post_processing/compressed_data/' + world + '/'):
        print('no data for world ' + world)
        sys.exit(1)
    training_data_file = '/home/whitesea/workspace/data_post_processing/processed_data/' + world + '/full_map_highway_training_data.npz'
    output_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'+ world + '/'
    if os.path.isfile(training_data_file):
        sys.exit(0)
    from toolkit import *
    from generate_visits import *
    data_path = '/home/whitesea/workspace/data_post_processing/compressed_data/' + world +'/'
    visits_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/visits.npy"
    if os.path.isfile(visits_file):
        visits = np.load(visits_file)
    else:
        visits = get_visits(data_path)
        np.save(visits_file, visits)

    visits = visits.astype(np.float32) / visits.max()
    visits = visits * 255.0

    im = imread(output_folder + 'map.pgm')
    im = Image.fromarray(im)
    im = im.resize((1024,1024),Image.ANTIALIAS)
    im = np.array(im)
    im = im.reshape(1024,1024,1)
    visits = Image.fromarray(visits)
    visits = visits.resize((1024,1024),Image.ANTIALIAS)
    visits = np.array(visits)
    visits = visits.reshape(1024,1024,1)
    out = np.concatenate([im,visits],axis=-1)
    out = out.astype(np.uint8)
    data = rotate_data(out)
    np.savez_compressed(training_data_file, data)
