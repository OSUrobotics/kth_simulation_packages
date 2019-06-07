#!/usr/bin/env python

import glob, sys, tqdm
import numpy as np


def pose2tuple(pose):
    return (pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

def remove_duplicates(data):
    reduced_list = []
    seen = []
    for d in data['data_points']:
        t = d['amcl_path']
        if t not in seen:
            seen.append(t)
            reduced_list.append(d)

    # data['data_points'] = reduced_list
    seen = []
    reduced_fails = []
    for d in data['recovery_locations']:
        seen.append(pose2tuple(d['amcl_pose']) + (d['recovery_behavior'],))
    s = set(seen)
    for p in s:
        reduced_fails.append(data['recovery_locations'][seen.index(p)])
    return_data = dict()
    return_data['data_points'] = reduced_list
    return_data['recovery_locations'] = reduced_fails
    return return_data

if __name__ == "__main__":
    files = glob.glob(sys.argv[1]+'*.npy')
    data = dict()
    data['data_points'] = []
    data['recovery_locations'] = []
    for file in tqdm.tqdm(files):
        try:
            point = np.load(file)
        except EOFError:
            continue
        point = point.item()
        data['data_points'] = data['data_points'] + point['data_points']
        data['recovery_locations'] = data['recovery_locations'] + point['recovery_locations']
    data = remove_duplicates(data)
    np.savez_compressed(sys.argv[2] + 'data.npz',data)
