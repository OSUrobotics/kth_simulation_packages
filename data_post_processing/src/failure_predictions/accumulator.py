#!/usr/bin/env python
import numpy as np

def process_label_file(file,window_size,step_size):
    labels = np.load(file)
    height,width = labels.shape
    

if __name__ == "__main__":
    world = sys.argv[1]
    window_size = float(sys.argv[2])
    step_size = float(sys.argv[3])

    label_file = '/home/whitesea/workspace/data_post_processing/processed_data/'+world + "/labels.npy"
