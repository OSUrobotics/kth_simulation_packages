import numpy as np
from scipy.misc import imread
import sys, os
from PIL import Image

if __name__ == "__main__":
    datapath = sys.argv[1]
    if os.path.exists(datapath + 'full_map_failure_rate_training_data.npy'):
        sys.exit()
    im = imread(datapath + 'map.pgm')
    im = Image.fromarray(im)
    im = im.resize((1024,1024),Image.ANTIALIAS)
    im = np.array(im)
    im = im.reshape(1024,1024,1)
    heatmap = np.load(datapath+'failure_rates.npy')
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((1024,1024),Image.ANTIALIAS)
    heatmap = np.array(heatmap)
    heatmap = heatmap.reshape((heatmap.shape[0],heatmap.shape[1],1))
    # visits = np.load(datapath+'visits.npy')
    # visits = visits / visits.max()
    # visits = visits * 255.0
    # visits = visits.astype(np.uint8)
    # visits = Image.fromarray(visits)
    # visits = visits.resize((1024,1024),Image.ANTIALIAS)
    # visits = np.array(visits)
    # visits = visits.reshape((visits.shape[0],visits.shape[1],1))
    # out = np.concatenate([im,visits,heatmap], axis=-1)
    out = np.concatenate([im,heatmap],axis=-1)
    out = out.astype(np.uint8)
    # np.save(datapath+'fully_convolutional_paths_data.npy',out)
    np.save(datapath+'full_map_failure_rate_training_data.npy',out)
