#!/usr/bin/env python
import numpy as np
import glob, sys, os, tqdm


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
    return_data = dict()
    return_data['data_points'] = reduced_list
    return return_data

if __name__ == "__main__":
    input_path = sys.argv[1]
    print(input_path)
    output_folder = sys.argv[2]
    files = glob.glob(input_path+'*.npy')
    data = dict()
    data['data_points'] = []
    for f in tqdm.tqdm(files):
        try:
            point = np.load(f)
        except EOFError:
            continue
        point = point.item()
        data['data_points'] = data['data_points'] + point['data_points']

    data = remove_duplicates(data)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.savez_compressed(output_folder + '/data.npz',data)
