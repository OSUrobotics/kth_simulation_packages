#!/usr/bin/env python

import sys, glob, yaml, random
import numpy as np
from scipy.misc import imread
from copy import deepcopy
from time import time
import cv2 as cv
from tqdm import tqdm

np.random.seed(int(time()))

def _get_window(im,res,y,x):
    left_bound = x + int(2.5/res)
    right_bound = x + int(7.5/res)
    upper_bound = y + int(2.5/res)
    lower_bound = y + int(7.5/res)
    return im[upper_bound:lower_bound,left_bound:right_bound]

def get_samples_wavefront(image, origin, res, sample_rate = .01, max_samples = 10000):

    ########### Inflate the image and set up dictionaries #################
    height, width = image.shape
    radius = int(.30/res)
    open_dict = dict()
    closed_dict = dict()
    img = deepcopy(image)
    x_org = -int(origin[0] / res)
    y_org = height + int(origin[1] / res)
    for i in range(width):
        for j in range(height):
            open_dict[(i,j)] = False
            closed_dict[(i,j)] = False
            if image[j,i] == 0:
                img = cv.circle(img,(i,j),radius,color=0,thickness = -1)
                open_dict[(x_org,y_org)] = True
                open_list = [(x_org,y_org)]

    ############# find all reachable pixels with a wavefront ################
    samples = [(x_org,y_org)]

    while open_list:
        sample = open_list.pop(0)
        x = sample[0]
        y = sample[1]
        closed_dict[(x,y)] = True
        samples.append((y,x)) #row,column order
        for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if  i < 0 or j < 0 or i >= width or j >= height or open_dict[(i,j)]:
                pass
            else:
                if img[j,i] == 0:
                    open_dict[(i,j)] = True
                else:
                    open_dict[(i,j)] = True
                    open_list.append((i,j))

    ##### Downsample the reachable space proportional to the sample rate #####
    proportional_samples = int(len(samples)*sample_rate)
    n_samples = max_samples
    if proportional_samples < max_samples:
        n_samples = proportional_samples
    downsamples = list(random.sample(samples,n_samples))

    ############## plot the downsamples space for debugging ################
    # testim = deepcopy(img)
    # testim = cv2.cvtColor(testim,cv2.COLOR_GRAY2RGB)
    # for goal in downsamples:
    #     testim[goal[1],goal[0]] = (0,255,0)
    # cv2.imwrite('downsampled.png',testim)

    return downsamples

def generate_data(filepath,samples,step):

    im = imread(filepath+'map.pgm')
    height,width = im.shape
    config = yaml.load(open(filepath+'map.yaml','r'))
    res = config['resolution']
    origin = config['origin']
    try:
        visits = np.load(filepath + 'visits.npy')
    except:
        raise IOError('No visits file found in {}'.format(filepath))

    candidates = get_samples_wavefront(im, origin,res)
    horizontal_edge = np.ones((int(5/res),width)) * 205
    vertical_edge = np.ones((height+(int(10/res)),int(5/res))) * 205
    im = np.concatenate((im,horizontal_edge),axis=0)
    im = np.concatenate((horizontal_edge,im),axis=0)
    im = np.concatenate((vertical_edge,im),axis=1)
    im = np.concatenate((im,vertical_edge),axis=1)

    data = []
    h,w = visits.shape
    max_y = 0
    for i in tqdm(range(samples)):
        if len(candidates) < 1:
            break
        ind = np.random.choice(len(candidates))
        y,x = candidates[ind]
        candidates.remove((y,x))
        Y = visits[y:y+step,x:x+step].sum()
        if Y == 0:
            i = i - 1
            continue
        if Y > max_y:
            max_y = Y
        X = _get_window(im,res,y,x) / 255.0
        data.append([X,Y])
    for i in range(len(data)):
        data[i][1] = data[i][1] / max_y

    return data

if __name__ == "__main__":
    filepath = sys.argv[1]
    data = generate_data(filepath,1000,8)
    np.save(filepath + 'highway_data.npy',data)
