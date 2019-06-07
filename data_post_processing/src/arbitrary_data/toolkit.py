#!/usr/bin/env python

import sys, yaml, time
import numpy as np
from scipy.misc import imread,imsave

# Returns a window around a given x and y pixel. window size in meters, resolution
# in meters/pixel
def _get_window(im,y,x,window_size,res):
    left_bound = x
    right_bound = x + int(float(window_size)/float(res))
    upper_bound = y
    lower_bound = y + int(float(window_size)/float(res))
    return im[upper_bound:lower_bound,left_bound:right_bound]

def _extend_image(im,window_size,res,val = 205):
    height,width = im.shape
    horizontal_edge = np.ones((int((window_size*.5)/res),width)) * val
    vertical_edge = np.ones((height+(int(window_size/res)),int((window_size*.5)/res))) * val
    new_im = np.concatenate((im,horizontal_edge),axis=0)
    new_im = np.concatenate((horizontal_edge,new_im),axis=0)
    new_im = np.concatenate((vertical_edge,new_im),axis=1)
    new_im = np.concatenate((new_im,vertical_edge),axis=1)
    return new_im

def generate_relative_training_data(filepath,samples=500,window_size = 10):
    seed = int(time.time())
    np.random.seed(seed)
    try:
        relative_values = np.load(filepath + 'relative_values.npy')
    except:
        print('no relative_values.npy file found in ' + filepath)
        sys.exit(1)

    height,width = relative_values.shape
    config = yaml.load(open(filepath+ 'map.yaml','rb'))
    res = config['resolution']
    im = imread(filepath + 'map.pgm')

    data = []
    diff = int((window_size) / res)
    upper_x = width - diff
    upper_y = height - diff
    i = 0
    while i < samples:
        x = np.random.randint(0,upper_x)
        y = np.random.randint(0,upper_y)
        x_map = _get_window(im,y,x,window_size,res)
        if x_map.mean() == x_map.max():
            pass
        else:
            x_map = x_map.reshape(x_map.shape[0],x_map.shape[1],1)
            x_heatmap = _get_window(relative_values,y,x,window_size,res)
            x_heatmap = x_heatmap.reshape(x_heatmap.shape[0],x_heatmap.shape[1],1)
            point = np.concatenate([x_map,x_heatmap],axis=-1)
            data.append(point)
            i += 1

    return np.array(data)

def generate_failure_rate_training_data(output_folder, samples=500, window_size=10):
