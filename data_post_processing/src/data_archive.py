#!/usr/bin/env python
import numpy as np
import os, sys, time

def import_data(file):
    data = np.load(file)
    data = data.item()
    return data

def combine_data(data1,data2):
    data1['data_points'] = data1['data_points'] + data2['data_points']
    data1['recovery_locations'] = data1['recovery_locations'] + data2['recovery_locations']
    return data1

def main(args):
    dirname = os.path.dirname(os.path.dirname(__file__))
    world = os.path.basename(os.path.dirname(args[1]))
    if not os.path.exists(dirname+'/data/'+world):
        os.makedirs(dirname+'/data/'+world)
    data = import_data(args[1])
    datafile = dirname + "/data/" + world + "/"+time.strftime("%d_%m_%Y_%H_%M_%S")+".npy"

    np.save(datafile,data)

if __name__ == "__main__":
    args = sys.argv
    main(args)
