#!/usr/bin/env python
import cv2, sys, yaml
import numpy as np
from scipy.misc import imread
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_map(filename):
    im = imread(filename)
    return im

def load_visits_normalized(filename):
    data = np.load(filename)
    return data

def create_binned_heatmap(heatmap, binning_factor = 2):
    h,w = heatmap.shape
    new_im = np.zeros((h,w))
    for i in np.arange(0,h - binning_factor,binning_factor):
        for j in np.arange(0,w - binning_factor,binning_factor):
            new_im[i:i+binning_factor,j:j+binning_factor] = heatmap[i:i+binning_factor,j:j+binning_factor].sum()
    return new_im

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap(np.arange(cmap.N))
    mycmap[:,-1] = np.linspace(0, 1, cmap.N)
    mycmap = ListedColormap(mycmap)
    return mycmap

if __name__ == "__main__":
    filepath = sys.argv[1]
    im = load_map(filepath + 'map.pgm')
    data = load_visits_normalized(filepath + 'visits.npy')
    heatmap = create_binned_heatmap(data,binning_factor=8)
    mycmap = transparent_cmap(plt.cm.plasma)
    fig,ax = plt.subplots(1,1)
    ax.imshow(im,cmap='gray')
    cb = ax.imshow(heatmap,cmap=plt.cm.plasma)
    plt.savefig(filepath+'visits.png')
