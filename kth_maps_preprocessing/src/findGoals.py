#!/usr/bin/env python
# findGoals.py
# Input: location of map.pgm and map.yaml file used in ROS navigation
# Output: Yaml configuration file with a set of coordinates in the world that
# can view most of the world unobstructed
# TODO: Add in exception handling, add to pipeline to for use in ROS
"""
MIT License
Copyright (c) 2018 Austin Whitesell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from time import sleep
import yaml, os, mdptoolbox, random, cv2, sys
from copy import deepcopy
import numpy as np
import random
from skimage.draw import line
from scipy.misc import imread
# from scipy.spatial.distance import euclidean

res = .05
origin = None

############################# load_map definition #######################
# Load map takes in two arguments, both of which are filepaths
# The image argument expects the file path to a pgm image which represents
# the map
# The config argument expects the file path to a yaml file which defines the
# resolution of the file, and the origin defined by the coordinates of
# lower left corner o the image
#
# The function returns the image as a numpy array, the resolution as a floating
# point value, and the origin as a tuple of the coordinates of the lower left
# corner of the image, in meters
##########################################################################
def load_map(image, config):
    ########## Load the configuration from the map.yaml file ##############
    generator = yaml.load_all(open(config, 'r'))
    config = None
    for thing in generator:
        config = thing

    res = config['resolution']
    origin = config['origin']

    ################## Read in the image ##################################
    img = imread(image)
    return img, res, origin

############################# get_samples_wavefront definition ###############
# get_samples_wavefront starts at (0,0) position as defined by the origin,
# which defines the coordinates of the lower left corner of the map, and
# finds all reachable points in the map, and then downsamples this list
# to approximate the reachable space in the map.
#
# The image argument expects a grayscale numpy array with values from 0 to 255
# that represents the map
#
# The res argument expects a floating point value that represents the number of
# meters per pixel
#
# The origin argument expects a tuple that defines the coordinates of the lower
# left pixel of the map (x,y) in meters
#
# The sample_rate argument expects a value between 0 and 1 which defines the
# percentage of total reachable pixels should be included in the returned
# samples list
##############################################################################
def get_samples_wavefront(image, sample_rate = .01, max_samples = 10000):

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
                img = cv2.circle(img,(i,j),radius,color=0,thickness = -1)
                open_dict[(x_org,y_org)] = True
                open_list = [(x_org,y_org)]

    ############# find all reachable pixels with a wavefront ################
    samples = [(x_org,y_org)]

    while open_list:
        sample = open_list.pop(0)
        x = sample[0]
        y = sample[1]
        closed_dict[(x,y)] = True
        samples.append(sample)
        for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if open_dict[(i,j)] or i < 0 or j < 0 or i >= width or j >= height:
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

############################ get_cover definition #############################
# get_cover finds a set of goals that ensures that all observable points
# are visible from the goals. It returns a list of tuples of the metric
# cartesian coordinates for the observer to visit
#
# The image argument expects a grayscale numpy array with values from 0 to 255
# that represents the map
#
# The res argument expects a floating point value that represents the number of
# meters per pixel
#
# The samples argument expects a list of tuples, describing the pixels to be
# to be observed, in the format (column, row). Each sample must exist in
# the image, and accessible through the reference image[row,col]
#
# The origin argument expects a tuple that defines the coordinates of the lower
# left pixel of the map (x,y) in meters
##############################################################################
def get_cover(image, samples):
    ################# inflate the image ##########################
    height, width = image.shape
    radius = int(.7 / res)
    laser_range = int(10/res)
    img = deepcopy(image)
    goal_image = deepcopy(image)
    for i in range(height):
        for j in range(width):
            if image[i,j] == 0:
                img = cv2.circle(img, (j,i), radius, color = 0, thickness = -1)

    goal_image = cv2.cvtColor(goal_image,cv2.COLOR_GRAY2RGB)
    ################## Generate the graph of line of sight ###############
    visibility_img = np.zeros((height,width))
    graph = dict()
    open = []
    for i in range(len(samples)):
        graph[samples[i]] = []
        open.append(samples[i])
        goal_image[samples[i][1],samples[i][0]] = (255,0,0)
        visibility_img[samples[i][1],samples[i][0]] = 1
    for i in range(len(samples)):
        mask = np.zeros((height,width))
        x1,y1 = samples[i]
        mask = cv2.circle(mask,(x1,y1),radius = laser_range, color = 1, thickness = -1)
        out = np.logical_and(visibility_img,mask)
        neighbors = np.where(out)
        for k in range(len(neighbors[0])):
            y2,x2 = (neighbors[0][k],neighbors[1][k])
            rr,cc = line(y1,x1,y2,x2)
            if image[rr,cc].all():
                graph[(x1,y1)].append((x2,y2))
                graph[(x2,y2)].append((x1,y1))
        visibility_img[y1,x1] = 0

        # for j in range(i,len(samples)):
        #     sample1 = samples[i]
        #     sample2 = samples[j]
        #     dist = euclidean(sample1,sample2)
        #     if dist <= laser_range:
        #         rr,cc = line(sample1[1],sample1[0],sample2[1],sample2[0])
        #         if image[rr,cc].all():
        #             graph[sample1].append(sample2)
        #             graph[sample2].append(sample1)
    print "visibility found"


    ########################################################################
    # Using a greedy approach, add goals that have line of sight to the most
    # other points, until all points can be observed
    ########################################################################
    goals = []
    while open:
        biggest = 0
        goal = None
        for i in open:
            if img[i[1],i[0]] == 0:
                graph[i] = []
            elif len(graph[i]) > biggest:
                biggest = len(graph[i])
                goal = i
        if goal == None:
            break
        goals.append(goal)
        open.remove(goal)
        visited = [goal]
        for j in graph[goal]:
            try:
                visited.append(j)
                open.remove(j)
            except:
                pass
        graph[goal] = []
        for j,i in enumerate(graph):
            for k in visited:
                try:
                    graph[i].remove(k)
                except:
                    pass

    ######## Generate an image showing goals for debugging ####################
    x_org = -int(origin[0] / res)
    y_org = height + int(origin[1] / res)
    goal_image = cv2.circle(goal_image,(x_org,y_org),int(.3/res),color=(0,0,255), thickness = -1)
    for goal in goals:
        goal_image = cv2.circle(goal_image,(goal[0],goal[1]),int(.3/res),color=(255,0,0), thickness = -1)
    # cv2.imwrite('test.png',goal_image)

    ####### Convert the goals into cartesian coordinates for navigation #######
    for i in range(len(goals)):
        goals[i] = [goals[i][0] * res + origin[0], (height - goals[i][1]) * res + origin[1]]

    return goals, goal_image

############################ write_goals definition ###########################
# write_goals writes out a yaml file defining the coordinates in (x,y,z) and
# a quaternion orientation for an observer to observe the whole reachable space.
# The yaml file will be stored locally within kth_maps_preprocessing package
# under the subdirectory goals
#
# The goals argument expects a list of tuples in (x,y) coordinates, in meters
#
# The name argument expects a string to use as the name of the folder to contain
# the goals.yaml file
###############################################################################
def write_goals(goals,name,image):
    ################## Get the path to the goals folder #######################
    dirname = os.path.dirname(os.path.dirname(__file__))
    dirname = os.path.join(dirname,'goals/'+name+'/')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Convert the goals to dictionaries for yaml writing and add the no-rotation
    # orientation as a quaternion
    navigation_goals = []
    i = 0
    for goal in goals:
        name = 'goal_' + str(i)
        pose = dict(
            name = name,
            position = dict(
                x = goal[0],
                y = goal[1],
                z = 0,
            ),
            orientation = dict(
                x = 0,
                y = 0,
                z = 0,
                w = 1.0,
            )
        )
        navigation_goals.append(pose)
        i = i+1

    ################### Write the goals to the goalfile ######################
    with open(dirname + 'goals.yaml', 'w') as outfile:
        yaml.dump(navigation_goals, outfile, default_flow_style=False)

    cv2.imwrite(dirname+'/goals.png',image)

def main(args):
    global origin
    try:
        name = args[1].split('/')[-2]
        dirname = os.path.dirname(os.path.dirname(__file__))
        filename = os.path.join(dirname,'goals/'+name+'/goals.yaml')
        if os.path.exists(filename):
            print "Goals already exist for this map: " + name
        else:
            mapfile = args[1] + 'map.pgm'
            confile = args[1] + 'map.yaml'
            sample_rate = .02
            max_samples = 10000
            img, res, origin = load_map(mapfile,confile)
            print 'map loaded'
            samples = get_samples_wavefront(img,sample_rate,max_samples)
            print 'samples acquired'
            goals,goal_im = get_cover(img,samples)
            print 'goals calculated'
            write_goals(goals,name,goal_im)
            print 'goals written'
    except:
        print "Error processing file " + args[1]

if __name__ == "__main__":
    main(sys.argv)

# def get_samples_rrt(image, res, origin):
#
#     # inflate the image
#     height, width = image.shape
#
#     ## Rapidly Exploring Random tree to generate samples
#     x_org = -int(origin[0] / res)
#     y_org = height + int(origin[1] / res)
#     xs = range(width)
#     ys = range(height)
#     samples = [(x_org,y_org)]
#     i = 0
#     # testim = deepcopy(image)
#     # testim = cv2.cvtColor(testim,cv2.COLOR_GRAY2RGB)
#     # testim[(y_org,x_org)] = (0,255,0)
#     while i < 3000:
#         try:
#             sample = (random.sample(xs,1)[0],random.sample(ys,1)[0])
#             min = float('inf')
#             nearest = None
#             # cv2.imshow('test',testim)
#             # cv2.waitKey(3)
#             for vert in samples:
#                 dist = euclidean(vert,sample)
#                 if dist < min:
#                     min = dist
#                     nearest = vert
#             x = nearest[0] + int(10 * (sample[0] - nearest[0]) / euclidean(nearest,sample))
#             y = nearest[1] + int(10 * (sample[1] - nearest[1]) / euclidean(nearest,sample))
#             rr,cc = line(nearest[1],nearest[0],y,x)
#             if image[rr,cc].all():
#                 samples.append((x,y))
#                 i = i + 1
#                 # testim[rr,cc] = (0,255,0)
#         except:
#             pass
#     # cv2.destroyAllWindows()
#
#
#     return samples
