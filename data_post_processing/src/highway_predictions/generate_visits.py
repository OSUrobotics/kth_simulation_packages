#!/usr/bin/env python

#General Packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Packages to make life easier
from tqdm import tqdm, trange
import yaml, glob, sys, os
from skimage.draw import line
from scipy.misc import imread


def interpolate_data(last_pt, pt):
    rr,cc = line(last_pt[0],last_pt[1],pt[0],pt[1])
    points = zip(rr,cc)
    return points

def process_file(file):
    data = np.load(file)
    data = data['arr_0']
    data = data.item()
    return data

def get_visits(filepath):
    im = imread(filepath + 'map.pgm')
    config = yaml.load(open(filepath + 'map.yaml','r'))
    res = config['resolution']
    origin = config['origin']
    x_org = -origin[0]
    y_org = -origin[1]
    h,w = im.shape
    files = glob.glob(filepath+'*.npz')
    visits = np.zeros((h,w))
    for file in tqdm(files):
        try:
            data = process_file(file)
        except EOFError:
            continue
        for point in data['data_points']:
            if len(point['amcl_path']) > 500:
                continue
            for i in range(len(point['amcl_path']) - 1):
                start = point['amcl_path'][i]
                end = point['amcl_path'][i+1]
                if end == start:
                    pass
                else:
                    x0 = int((start.position.x + x_org) / res)
                    y0 = h - int((start.position.y + y_org) / res)
                    x1 = int((end.position.x + x_org) / res)
                    y1 = h - int((end.position.y + y_org) / res)
                    cells = interpolate_data((y0,x0),(y1,x1))
                    for cell in cells:
                        visits[cell[0],cell[1]] += 1
    # remove outliers
    # for i in range(h):
    #     for j in range(w):
    #         if visits[i,j] > 300:
    #             visits[i,j] = 300
    return visits

if __name__ == "__main__":
    filepath = sys.argv[1]
    output_path = sys.argv[2]
    if os.path.exists(output_path + 'visits.npy'):
        sys.exit()
    visits = get_visits(filepath)
    np.save(output_path+'visits.npy',visits)
