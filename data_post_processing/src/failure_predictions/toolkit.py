#!/usr/bin/env python
# Functions used for processing data from the navigation failure learning project
import numpy as np
import glob,yaml,random,tqdm, Image, os, sys, guppy, time, imutils
import cv2
from copy import deepcopy
from math import pi, exp
from scipy.interpolate import griddata
from scipy.misc import imread, imsave
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.draw import line

# Takes in the filepath of an numpy file in the same shape of the map, where the
# values of the cells range from 0 to 1, with higher values indicating a higher
# probability that the cell belongs to a failure location
# returns another numpy array of the same size of the map, where each cell either
# the value 0 or 2, where 2 indicates it is failure location
def generate_failure_labels(blurred_data_file,threshold=40):
    blurred_data = np.load(blurred_data_file)
    labels = deepcopy(blurred_data)
    max = labels.max()
    labels = labels * (255/max)
    labels = labels.astype(int)
    labels = (labels > threshold) * 2
    return labels

# Takes in a folder as an argument, and looks at all .npy files
# and assumes they are raw data from the simulator. It will also
# assume that the map.pgm and map.yaml file from ros are in that directory
# This function will identify all places the robot visited during simulation
# It will return a numpy array the same size as the map, with the values 0 or 1.
# 0 indicates the robot did not visit the cell, and 1 indicates it did
def generate_visited_labels(filepath):
    generator = yaml.load_all(open(filepath+"map.yaml",'r'))
    config = None
    for thing in generator:
        config = thing

    res = config['resolution']
    origin = config['origin']

    im = imread(filepath+"map.pgm")
    height,width = im.shape

    x_org = -origin[0]
    y_org = -origin[1]

    visited = set()

    files = glob.glob(filepath+'*.npy')
    for file in files:
        try:
            data = np.load(open(file,'rb'))
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for point in data['data_points']:
            for a_pose in point['amcl_path']:
                x = int((a_pose.position.x + x_org) / res)
                y = height - int((a_pose.position.y + y_org) / res)

                visited.add((x,y))
    labels = np.zeros((height,width))
    for x,y in visited:
        labels[y,x] = 1
    return labels

# Returns a window around a given x and y pixel.
# Adds unknown space if the image would run off the edge
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

# Takes in a folder where it expects three files: labels.npy, map.pgm, and map.yaml
# labels.npy is expected to be of the same size as map.pgm, and contains the values
# 1.0 for a location that the robot visited but did not fail, and the value 2.0
# for a location where the robot failed.
# It also takes in the optional argument samples, which is the maximum number
# of samples of failures and successes that will be returned, though it will
# always return a balanced set. The output is an array of shape n,2. n is the
# the total number of samples returned. Each datapoint is shaped [image,label],
# where the image is a 5m x 5m window from the map, and label is 0 or 1, 0 indicating
# it is not a failure location, and 1 indicating that it is a failure location
def generate_training_data(filepath,samples=500, window_size = 5.0):
    # hp = guppy.hpy()
    im = imread(filepath+'map.pgm')
    height,width = im.shape
    config = yaml.load(open(filepath+'map.yaml','r'))
    res = config['resolution']
    try:
        labels = np.load(filepath+'labels.npy')
    except:
        raise IOError('No labels file found in {}'.format(filepath))

    data = []
    h,w = labels.shape
    lable_tuples = []
    success_data = []
    failure_data = []
    for i in range(h):
        for j in range(w):
            if labels[i,j] == 1.0:
                success_data.append((i,j))
            elif labels[i,j] == 2.0:
                failure_data.append((i,j))
    output = []
    num = min([len(failure_data),len(success_data),samples])

    # Add unknown data to edges of the map to ensure no running off edges
    extended_image = _extend_image(im, window_size,res)
    for i in range(num):
        ind = np.random.choice(len(failure_data))
        fail_y,fail_x = failure_data[ind]
        del failure_data[ind]
        fail_window = _get_window(extended_image,fail_y,fail_x, window_size, res)
        output.append([fail_window,labels[fail_y,fail_x] - 1])
        ind = np.random.choice(len(success_data))
        y,x = success_data[ind]
        del success_data[ind]
        succeed_window = _get_window(extended_image,y,x,window_size, res)
        output.append([succeed_window,labels[y,x] - 1])
    return np.array(output)

def generate_accumulated_training_data(filepath,samples=500,window_size = 5.0,threshold = 0.5,draw = False):
    try:
        accumulated = np.load(filepath+'accumulated.npy')
    except:
        print('no accumulated.npy file found in '+ filepath)
        sys.exit(1)
    height,width = accumulated.shape
    config = yaml.load(open(filepath + 'map.yaml','rb'))
    res = config['resolution']
    im = imread(filepath + 'map.pgm')
    data = []
    accumulated = _extend_image(accumulated,window_size,res,val = 0.0)
    accumulated = accumulated / float(accumulated.max())
    accumulated = accumulated >= threshold
    reachable = get_samples_wavefront(im,config['origin'],res,sample_rate = .5)
    im = _extend_image(im,window_size,res)
    if draw:
        color_im = imread(filepath + 'map.pgm')
        color_im = cv2.cvtColor(color_im,cv2.COLOR_GRAY2RGB)
    fail_window_centers = []
    success_window_centers = []
    for i,j in reachable:
        if _get_window(accumulated,i,j,window_size,res).any():
            fail_window_centers.append((i,j))
        else:
            success_window_centers.append((i,j))
    num = min([len(fail_window_centers),len(success_window_centers),samples])
    for i in range(num):
        ind = np.random.choice(len(fail_window_centers))
        f_y,f_x = fail_window_centers[ind]
        del fail_window_centers[ind]
        fail_window = _get_window(im,f_y,f_x,window_size,res)
        ind = np.random.choice(len(success_window_centers))
        y,x = success_window_centers[ind]
        del success_window_centers[ind]
        succeed_window = _get_window(im,y,x,window_size,res)
        data.append([fail_window,1])
        data.append([succeed_window,0])
        if draw:
            cv2.rectangle(color_im,(f_x - int(float(window_size/res) / 2),f_y - int(float(window_size/res) / 2)),(f_x + int(float(window_size/res) / 2),f_y + int(float(window_size/res) / 2)),(255,0,0),1)
            cv2.rectangle(color_im,(x - int(float(window_size/res) / 2),y - int(float(window_size/res) / 2)),(x + int(float(window_size/res) / 2),y + int(float(window_size/res) / 2)),(0,255,0),1)
    if draw:
        return (np.array(data),color_im)
    else:
        return np.array(data)

def generate_accumulated_data(output_folder,window_size=5,stride=.25):
    config = yaml.load(open(output_folder + 'map.yaml','rb'))
    res = config['resolution']
    try:
        labels = np.load(output_folder+'failure_labels.npy')
    except:
        print('Failure labels file not found in ' + output_folder)
    extended_labels = _extend_image(labels,window_size,res,val=0.0)
    accumulated = np.zeros(extended_labels.shape)
    height,width = accumulated.shape
    step = int(stride / res)
    window = int(window_size / res)
    y = 0
    while y <= (height - window):
        x = 0
        while x <= (width - window):
            if extended_labels[y:y+window,x:x+window].sum() > 0:
                accumulated[y:y+window,x:x+window] += 1
            x += step
        y += step
    accumulated = accumulated[int(window/2):-int(window/2),int(window/2):-int(window/2)]
    return accumulated

# returns a list of y,x pixel coordinates reachable by the robot. Either 10000,
# or 1 percent of all reachable locations, whichever is less.
def get_samples_wavefront(image, origin, res, sample_rate = .01, max_samples = 10000, rad = 0.3):

    ########### Inflate the image and set up dictionaries #################
    height, width = image.shape
    radius = int(rad/res)
    open_dict = dict()
    closed_dict = dict()
    img = deepcopy(image)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    x_org = -int(origin[0] / res)
    y_org = height + int(origin[1] / res)
    for i in range(width):
        for j in range(height):
            open_dict[(i,j)] = False
            closed_dict[(i,j)] = False
            if not image[j,i].any():
                img = cv2.circle(img,(i,j),radius,color=(0,0,0),thickness = -1)
                open_dict[(x_org,y_org)] = True
                open_list = [(x_org,y_org)]

    ############# find all reachable pixels with a wavefront ################
    samples = [(y_org,x_org)]

    while open_list:
        sample = open_list.pop(0)
        x = sample[0]
        y = sample[1]
        closed_dict[(x,y)] = True
        samples.append((y,x)) #row,column order
        img[y,x] = (255,0,0)
        for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if  i < 0 or j < 0 or i >= width or j >= height or open_dict[(i,j)]:
                pass
            else:
                if not img[j,i].any():
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
    # plt.imshow(img)
    # plt.show()

    ############## plot the downsamples space for debugging ################
    # testim = deepcopy(img)
    # testim = cv2.cvtColor(testim,cv2.COLOR_GRAY2RGB)
    # for goal in downsamples:
    #     testim[goal[1],goal[0]] = (0,255,0)
    # cv2.imwrite('downsampled.png',testim)

    return downsamples

# takes in a path to a folder, which is assumed to have the map.pgm and map.yaml
# files from the ros navigation stack. The function will return a list of 1000
# 5m x 5m windows in the map to be used to predict if the center pixel of the window
# is a location where a robot would fail to navigate. each datapoint in the list
# contains the window, and a tuple (y,x) which corresponds to the pixel location
# in the original map where the window is centered.
def generate_prediction_inputs(filepath, window_size=5.0):
    im = imread(filepath+'map.pgm')
    height,width = im.shape
    config = yaml.load(open(filepath+'map.yaml','r'))
    res = config['resolution']
    origin = config['origin']

    pixel_tuples = get_samples_wavefront(im,origin,res)
    # Add unknown data to edges of the map to ensure no running off edges

    data = []
    extended_image = _extend_image(im,5,res)
    for i in range(1000):
        ind = np.random.choice(len(pixel_tuples))
        tup = pixel_tuples[ind]
        data.append([_get_window(extended_image,tup[0],tup[1], window_size, res)/255.,tup])
        del pixel_tuples[ind]
    return data

def import_data(filename):
    data = np.load(filename)
    data = data.item()
    return data

def load_map(filename, config):
    global res
    global origin
    generator = yaml.load_all(open(config, 'r'))
    config = None
    for thing in generator:
        config = thing

    res = config['resolution']
    origin = config['origin']

    img = imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def plot_paths(data,img):
    global res
    global origin
    height, width, channels = img.shape

    x_org = -origin[0]
    y_org = -origin[1]
    for point in data['data_points']:
        for a_pose in point['amcl_path']:
            x = int((a_pose.position.x + x_org) / res)
            y = height - int((a_pose.position.y + y_org) / res)

            img[y,x] = (255,0,0)

def label_failures(datapath):
    config = yaml.load(open(datapath+"map.yaml",'rb'))
    res = config['resolution']
    origin = config['origin']

    im = imread(datapath+"map.pgm")
    height,width = im.shape

    x_org = -origin[0]
    y_org = -origin[1]

    visited = set()

    files = glob.glob(datapath+'*.npy')
    for file in files:
        try:
            data = np.load(open(file,'rb'))
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for point in data['recovery_locations']:
            x = int((point["amcl_pose"].position.x + x_org) / res)
            y = height - int((point["amcl_pose"].position.y + y_org) / res)
            # img[y,x] = (0,0,255)
            if point["recovery_behavior"] == "rotate_recovery":
                visited.add((y,x))
    labels = np.zeros((height,width))
    for y,x in visited:
        labels[y,x] = 1
    return labels

def plot_points_raw(datafolder, resolution_scaling=2.0):
    img = imread(datafolder + 'map.pgm')
    config = yaml.load(open(datafolder + 'map.yaml','rb'))
    origin = config['origin']
    res = config['resolution']
    height, width = img.shape
    scaled_res = resolution_scaling * res
    data = dict()
    data['recovery_locations'] = []
    data['data_points'] = []
    files = glob.glob(datafolder+'*.npy')
    for file in files:
        point = np.load(open(file,'rb'))
        point = point.item()
        data['recovery_locations'] += point['recovery_locations']
        data['data_points'] += point['data_points']
    x_org = -origin[0]
    y_org = -origin[1]
    fails = np.zeros((int(height/resolution_scaling),int(width/resolution_scaling)))

    h,w = fails.shape
    for point in data['recovery_locations']:
        x = int((point["amcl_pose"].position.x + x_org) / scaled_res)
        y = h - int((point["amcl_pose"].position.y + y_org) / scaled_res)
        # img[y,x] = (0,0,255)
        if point["recovery_behavior"] == "rotate_recovery":
            fails[y,x] += 1.0
    return fails

def make_heatmap(datapath,robot_radius=.3):
    mapfile = datapath + 'map.pgm'
    mapconfig = datapath + 'map.yaml'
    pdf = lambda ((x, y), ux, uy, sx, sy) : (1/(2*pi*sx*sy))*exp((-1/2)*((((x-ux)**2)/(sx**2)) + (((y-uy)**2)/(sy**2))))
    map_params = yaml.load(open(mapconfig, 'rb'))
    I = Image.open(mapfile)
    w,h = I.size
    files = glob.glob(datapath + '*.npy')
    origin = map_params['origin']
    x_org = -origin[0]
    y_org = -origin[1]
    resolution = map_params['resolution']
    std_dev = int(robot_radius/resolution)
    fail_locations = []
    fails = []
    filtered = []
    for file in files:
        try:
            data = np.load(open(file,'rb'))
        except EOFError:
            print('error with ' + file)
            continue
        data = data.item()
        for failure in data['recovery_locations']:
            if failure['recovery_behavior'] == 'rotate_recovery':
                fails.append(failure)

    for failure in fails:
        if failure not in filtered:
            filtered.append(failure)

    for failure in filtered:
        x = failure["amcl_pose"].position.x - origin[0]
        y = failure["amcl_pose"].position.y - origin[1]
        x_coord = int(x / resolution)
        y_coord = h - int(y / resolution) - 1
        fail_locations.append((x_coord,y_coord))

    heatmap = np.zeros((h,w))
    s = 0.0
    for k in tqdm.tqdm(range(len(fail_locations))):
        x,y = fail_locations[k]
        update = np.array(map(pdf, [((j,i),x,y,std_dev,std_dev) for i in range(h) for j in range(w)])).reshape(h,w)
        s = s + update.sum()
        heatmap = heatmap + update

    for i in range(w):
        for j in range(h):
            heatmap[j,i] = heatmap[j,i] / s

    return heatmap

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap(np.arange(cmap.N))
    mycmap[:,-1] = np.linspace(0, 1, cmap.N)
    mycmap = ListedColormap(mycmap)
    return mycmap

def make_image(path_to_map,path_to_data):
    # Import image and get x and y extents
    I = Image.open(path_to_map+'map.pgm')
    w, h = I.size
    y, x = np.mgrid[0:h, 0:w]
    mycmap = transparent_cmap(plt.cm.plasma)
    #Plot image and overlay colormap
    fig, ax = plt.subplots(1, 1)
    ax.imshow(I, cmap='gray')
    heatmap = make_heatmap(path_to_map,path_to_data,robot_radius=.3)
    np.save(path_to_data + 'blurred_fails.npy',heatmap)
    cb = ax.imshow(heatmap,cmap=mycmap)
    plt.colorbar(cb)
    plt.savefig(path_to_data + 'heatmap.png')

def get_inputs_from_file(filename):
    data = np.load(filename)
    inputs = list(data[:,0])
    labels = list(data[:,1])
    return inputs, labels

def generate_roc_curve_from_outputs(processed_data_folder,model_file,splitfile,datafile_pattern):
    model = load_model(model_file)
    files = []
    with open(splitfile) as f:
        flag = False
        for line in f.readlines():
            if flag:
                chunks = line.split('/')
                world = chunks[1]
                files.append(processed_data_folder + world + '/' + datafile_pattern)
            elif 'validation files' in line:
                flag = True
    inputs = []
    labels = []
    for f in files:
        new_set,labs = get_inputs_from_file(f)
        inputs = inputs + new_set
        labels = labels + labs
    inputs = np.array(inputs)
    inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2],1)
    predictions = model.predict_proba(inputs)
    TPR = []
    FPR = []
    for i in np.linspace(0,1,100):
        generated_labels = [j >= i for j in predictions]
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for k in range(len(generated_labels)):
            if generated_labels[k] and labels[k]:
                TP += 1
            if generated_labels[k] and not labels[k]:
                FP += 1
            if not generated_labels[k] and labels[k]:
                FN += 1
            if not generated_labels[k] and not labels[k]:
                TN += 1
        TPR.append(float(TP) / float(TP + FN))
        FPR.append(float(FP) / float(FP + TN))
    return((TPR,FPR, predictions))

def pose2tuple(pose):
    return (pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

def load_failure_locations(files):
    pose_tuples = []
    for file in tqdm.tqdm(files):
        try:
            data = np.load(open(file,'rb'))
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for failure in data['recovery_locations']:
            if failure['recovery_behavior'] == 'rotate_recovery':
                pose_tuples.append(pose2tuple(failure['amcl_pose']))
    pose_tuples = set(pose_tuples)
    pose_tuples = list(pose_tuples)
    return pose_tuples

def load_paths(files):
    paths = []
    for file in tqdm.tqdm(files):
        try:
            data = np.load(open(file,'rb'))
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for point in data['data_points']:
            path = []
            for pose in point['amcl_path']:
                path.append(pose2tuple(pose))
            paths.append(path)
    return paths

def load_paths_fails(files):
    fails = []
    paths = []
    for file in tqdm.tqdm(files):
        try:
            data = np.load(open(file,'rb'))
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for point in data['data_points']:
            path = []
            for pose in point['amcl_path']:
                path.append(pose2tuple(pose))
            paths.append(path)
        for failure in data['recovery_locations']:
            if failure['recovery_behavior'] == 'rotate_recovery':
                fails.append(pose2tuple(failure['amcl_pose']))
    paths = [list(x) for x in set(tuple(x) for x in paths)]
    fails = list(set(fails))
    return fails, paths

def load_paths_fails_compressed(files):
    fails = []
    paths = []
    for file in tqdm.tqdm(files):
        try:
            data = np.load(file,allow_pickle=True)
            data = data['arr_0']
            data = data.item()
        except EOFError:
            print('issue with ' + file)
        for point in data['data_points']:
            path = []
            for pose in point['amcl_path']:
                path.append(pose2tuple(pose))
            paths.append(path)
        for failure in data['recovery_locations']:
            if failure['recovery_behavior'] == 'rotate_recovery':
                fails.append(pose2tuple(failure['amcl_pose']))
    # paths = [list(x) for x in set(tuple(x) for x in paths)]
    # fails = list(set(fails))
    return fails, paths

def generate_failure_rate_heatmap_blurring(data_folder, resolution_scaling=2.0, threshold=1.0):
    files = glob.glob(data_folder+ '*.npz')
    im = imread(data_folder + 'map.pgm')
    config = yaml.load(open(data_folder+'map.yaml','rb'))
    res = config['resolution']
    scaled_res = resolution_scaling * res
    origin = config['origin']
    x_org = -origin[0]
    y_org = -origin[1]
    height,width = im.shape
    scaled_heatmap = np.zeros((int(height/resolution_scaling),int(width/resolution_scaling),1))
    h,w,c = scaled_heatmap.shape
    print(bcolors.OKGREEN + "LOADING DATA"+ bcolors.ENDC)
    pose_tuples, paths = load_paths_fails_compressed(files)
    print(bcolors.OKGREEN + "CALCULATING FAILURE COUNTS PER CELL"+ bcolors.ENDC)
    for pose in tqdm.tqdm(pose_tuples):
        x = int((pose[0] + x_org) / scaled_res)
        y = h - int((pose[1] + y_org) / scaled_res)
        if x == w:
            x -=1
        if y == h:
            y -= 1
        scaled_heatmap[y,x] += 1.0


    # find number of times paths cross through each cell, turn failures into rates
    paths_count = np.zeros((int(height/resolution_scaling),int(width/resolution_scaling),1))
    return_im = np.zeros((height,width))
    print(bcolors.OKGREEN + 'CALCULATING NUMBER OF CELL CROSSINGS'+ bcolors.ENDC)
    for path in tqdm.tqdm(paths):
        update = np.zeros((h,w,1))
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            if end == start:
                pass
            else:
                x0 = int((start[0] + x_org) / scaled_res)
                y0 = h - int((start[1] + y_org) / scaled_res)
                x1 = int((end[0] + x_org) / scaled_res)
                y1 = h - int((end[1] + y_org) / scaled_res)

                if x0 == w:
                    x0 -= 1
                if x1 == w:
                    x1 -= 1
                if y0 == h:
                    y0 -= 1
                if y1 == h:
                    y1 -= 1

                cells = interpolate_data((y0,x0),(y1,x1))

                # print (x0,y0,x1,y1)
                for cell in cells:
                    update[cell[0],cell[1]] = 1
        paths_count += update
    scaled_heatmap = (scaled_heatmap > threshold) * scaled_heatmap
    print(bcolors.OKGREEN + 'CALCULATING FAILURE RATE PER CELL'+ bcolors.ENDC)
    for i in range(h):
        for j in range(w):
            if paths_count[i,j] > 0:
                scaled_heatmap[i,j] = float(scaled_heatmap[i,j] / paths_count[i,j])

    open_list = []
    for i in range(height):
        for j in range(width):
            return_im[i,j] = scaled_heatmap[int(i/resolution_scaling),int(j/resolution_scaling)]
            if return_im[i,j] > 0.0 or np.random.random() < .005:
                open_list.append([i,j])
    # find reachable space
    # reachable = get_samples_wavefront(im, origin, res, sample_rate = 1.0, max_samples = float('inf'), rad=0.3)
    # vals = np.zeros((height,width))
    # update = np.zeros((height,width))
    # for i in reachable:
    #     vals[i[0],i[1]] = 1
    # Blurr failure points with value iteration
    # print(bcolors.OKBLUE + "PERFORMING VALUE ITERATION TO BLURR" + bcolors.ENDC)
    # for k in tqdm.tqdm(range(50)):
    #     if (k % 2) == 0:
    #         for i in range(height):
    #             for j in range(width):
    #                 # value iteration
    #                 qs = [return_im[i,j]]
    #                 for y,x in [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1),(i+1,j+1)]: #
    #                     try:
    #                         qs.append(.99*return_im[y,x])
    #                     except:
    #                         pass
    #                 return_im[i,j] = max(qs)
    #     else:
    #         for i in reversed(range(height)):
    #             for j in reversed(range(width)):
    #                 # value iteration
    #                 qs = [return_im[i,j]]
    #                 for y,x in [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1),(i+1,j+1)]: #
    #                     try:
    #                         qs.append(.99*return_im[y,x])
    #                     except:
    #                         pass
    #                 return_im[i,j] = max(qs)

    grid_x, grid_y = np.mgrid[0:height,0:width]
    points = np.array(open_list)
    values = []
    for i,j in points:
        values.append(return_im[i,j])
    values = np.array(values)
    print(bcolors.OKGREEN + 'BLURRING FAILURE RATES'+ bcolors.ENDC)

    return_im = griddata(points,values,(grid_x,grid_y),method='linear')
    # return_im = return_im / return_im.max()
    # return_im = (vals > 0) * return_im
    return_im = return_im*255.0
    return_im = return_im.astype(np.uint8)
    return return_im

def generate_failure_rate_heatmap(data_folder, resolution_scaling=2.0, threshold=1.0):
    files = glob.glob(data_folder+ '*.npz')
    im = imread(data_folder + 'map.pgm')
    config = yaml.load(open(data_folder+'map.yaml','rb'))
    res = config['resolution']
    scaled_res = resolution_scaling * res
    origin = config['origin']
    x_org = -origin[0]
    y_org = -origin[1]
    height,width = im.shape
    scaled_heatmap = np.zeros((int(height/resolution_scaling),int(width/resolution_scaling),1))
    h,w,c = scaled_heatmap.shape
    print(bcolors.OKGREEN + "LOADING DATA"+ bcolors.ENDC)
    pose_tuples, paths = load_paths_fails_compressed(files)
    print(bcolors.OKGREEN + "CALCULATING FAILURE COUNTS PER CELL"+ bcolors.ENDC)
    for pose in tqdm.tqdm(pose_tuples):
        x = int((pose[0] + x_org) / scaled_res)
        y = h - int((pose[1] + y_org) / scaled_res)
        if x == w:
            x -=1
        if y == h:
            y -= 1
        scaled_heatmap[y,x] += 1.0


    # find number of times paths cross through each cell, turn failures into rates
    paths_count = np.zeros((int(height/resolution_scaling),int(width/resolution_scaling),1))
    print(bcolors.OKGREEN + 'CALCULATING NUMBER OF CELL CROSSINGS'+ bcolors.ENDC)
    for path in tqdm.tqdm(paths):
        update = np.zeros((h,w,1))
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            if end == start:
                pass
            else:
                x0 = int((start[0] + x_org) / scaled_res)
                y0 = h - int((start[1] + y_org) / scaled_res)
                x1 = int((end[0] + x_org) / scaled_res)
                y1 = h - int((end[1] + y_org) / scaled_res)
                if x0 == w:
                    x0 -= 1
                if x1 == w:
                    x1 -= 1
                if y0 == h:
                    y0 -= 1
                if y1 == h:
                    y1 -= 1

                cells = interpolate_data((y0,x0),(y1,x1))
                # print (x0,y0,x1,y1)
                for cell in cells:
                    update[cell[0],cell[1]] = 1
        paths_count += update
    return_im = np.zeros((height,width))
    scaled_heatmap = (scaled_heatmap > threshold) * scaled_heatmap
    for i in range(h):
        for j in range(w):
            if paths_count[i,j] > 0:
                scaled_heatmap[i,j] = float(scaled_heatmap[i,j] / paths_count[i,j])

    for i in range(height):
        for j in range(width):
            y = int(i/resolution_scaling)
            x = int(j/resolution_scaling)
            if y == h:
                y -= 1
            if x == w:
                x -=1
            return_im[i,j] = scaled_heatmap[y,x]

    return_im = return_im*255.0
    return_im = return_im.astype(np.uint8)
    return return_im

def rotate_data(data):
    rotated_data = []
    print(bcolors.OKGREEN + 'ROTATING DATA'+ bcolors.ENDC)
    for i in np.arange(0,360,10):
        rotated_data.append(imutils.rotate(data,i))
    return np.array(rotated_data)

def generate_failure_rate_training_data(filepath,samples=500,window_size = 10):
    seed = int(time.time())
    np.random.seed(seed)
    try:
        failure_values = np.load(filepath + 'failure_rates.npy')
    except:
        print('no failure_rates.npy file found in ' + filepath)
        sys.exit(1)

    height,width = failure_values.shape
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
            x_heatmap = _get_window(failure_values,y,x,window_size,res)
            x_heatmap = x_heatmap.reshape(x_heatmap.shape[0],x_heatmap.shape[1],1)
            point = np.concatenate([x_map,x_heatmap],axis=-1)
            data.append(point)
            i += 1

    return np.array(data)

def generate_path_failure_training_data(filepath,samples=500,window_size = 10):
    seed = int(time.time())
    np.random.seed(seed)
    try:
        failure_values = np.load(filepath + 'failure_rates.npy')
        visit_values = np.load(filepath + 'visits.npy')
    except:
        print('no failure_rates.npy and/or visits.npy file found in ' + filepath)
        sys.exit(1)

    height,width = failure_values.shape
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
            x_fail_heatmap = _get_window(failure_values,y,x,window_size,res)
            x_fail_heatmap = x_fail_heatmap * 255
            x_fail_heatmap = x_fail_heatmap.astype(np.uint8)
            x_fail_heatmap = x_fail_heatmap.reshape(x_fail_heatmap.shape[0],x_fail_heatmap.shape[1],1)
            x_path_heatmap = _get_window(visit_values,y,x,window_size,res)
            x_path_heatmap = x_path_heatmap * 255
            x_path_heatmap = x_path_heatmap.astype(np.uint8)
            x_path_heatmap = x_path_heatmap.reshape(x_path_heatmap.shape[0],x_path_heatmap.shape[1],1)
            point = np.concatenate([x_map,x_path_heatmap,x_fail_heatmap],axis=-1)
            data.append(point)
            i += 1

    return np.array(data)

def interpolate_data(last_pt, pt):
    rr,cc = line(last_pt[0],last_pt[1],pt[0],pt[1])
    points = zip(rr,cc)
    return points

def draw_path_crossings(filepath):
    im = imread(filepath + 'map.pgm')
    config = yaml.load(open(filepath + 'map.yaml','r'))
    res = config['resolution']
    origin = config['origin']
    x_org = -origin[0]
    y_org = -origin[1]
    h,w = im.shape
    files = glob.glob(filepath+'*.npz')
    visits = np.zeros((h,w))
    for file in tqdm.tqdm(files):
        try:
            data = np.load(file)
            data = data['arr_0']
            data = data.item()
        except EOFError:
            continue
        for point in data['data_points']:
            update = np.zeros((h,w))
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
                        update[cell[0],cell[1]] = 1
            visits = visits + update
    # remove outliers
    # for i in range(h):
    #     for j in range(w):
    #         if visits[i,j] > 300:
    #             visits[i,j] = 300
    return visits

class model_tester:
    def __init__(self,model_file,type = 'standard'):
        self.type = type
        self.model = load_model(model_file)
        self.predictions = []
        self.labels = []
        self.predictions = []

    def generate_curves(self,folders,window_size = 5.0):
        if self.type == 'standard':
            return self._generate_curves_standard(folders,window_size)
        # elif type == 'accumulator':
        #     return self._generate_curves_accumulator(folders,window_size)
    def _generate_curves_standard(self,folders,window_size = 5.0):
        self._get_predictions(folders,window_size)
        TPR = []
        FPR = []
        for i in np.linspace(0,1,100):
            generated_labels = [j > i for j in self.predictions]
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            for k in range(len(generated_labels)):
                if generated_labels[k] and self.labels[k]:
                    TP += 1
                if generated_labels[k] and not self.labels[k]:
                    FP += 1
                if not generated_labels[k] and self.labels[k]:
                    FN += 1
                if not generated_labels[k] and not self.labels[k]:
                    TN += 1
            TPR.append(TP / (TP + FN))
            FPR.append(FP / (FP + TN))
        return((TPR,FPR))

    def save_predictions(self,filename):
        np.save(filename,self.predictions)

    def save_labels(self,filename):
        np.save(filename,self.labels)

    def _get_predictions(self,folders, window_size=5.0):
        for folder in tqdm.tqdm(folders):
            data = self._data_set_from_folder(folder, window_size)
            input = []
            labels = []
            for point in data:
                input.append(list(point[0]/255))
                labels.append(point[1])
            input = np.array(input)
            input = input.reshape(input.shape[0],input.shape[1],input.shape[2],1)
            predictions = self.model.predict(input)
            self.labels = self.labels + labels
            self.predictions = self.predictions + list(predictions[:,0])

    def _data_set_from_folder(self,folder_path, window_size=5.0):
        im  = imread(folder_path + "map.pgm")
        height,width = im.shape
        config = yaml.load(open(folder_path+'map.yaml','r'))
        res = config['resolution']
        origin = config['origin']
        x_org = -origin[0]
        y_org = -origin[1]

        datafiles = glob(folder_path + "/*.npy")
        labels = np.zeros((height,width))

        failure_locations = []
        success_locations = []
        for f in datafiles:
            data = np.load(open(f,'rb'))
            data = data.item()
            for failure in data['recovery_locations']:
                if failure['recovery_behavior'] == 'rotate_recovery':
                    x = int((failure["amcl_pose"].position.x + x_org) / res)
                    y = height - int((failure["amcl_pose"].position.y + y_org) / res)
                    labels[y,x] = 2
                    failure_locations.append((y,x))

            for point in data['data_points']:
                for a_pose in point['amcl_path']:
                    x = int((a_pose.position.x + x_org) / res)
                    y = height - int((a_pose.position.y + y_org) / res)
                    if not labels[y,x]:
                        labels[y,x] = 1
                        success_locations.append((y,x))

        output = []
        num = min([len(failure_locations),len(success_locations)])
        extended_image = _extend_image(im,window_size,res)
        for i in range(num):
            ind = np.random.choice(len(failure_locations))
            fail_y,fail_x = failure_locations[ind]
            del failure_locations[ind]
            output.append([_get_window(extended_image,fail_y,fail_x,window_size,res),labels[y,x] - 1])
            ind = np.random.choice(len(success_locations))
            y,x = success_locations[ind]
            del success_locations[ind]
            output.append([_get_window(extended_image,y,x,window_size,res),labels[y,x] - 1])
        return np.array(output)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
