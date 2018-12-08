#!/usr/bin/env python
# model_map_builder.py
# Input: An xml file which describes the single-level floorplan of a
# building or buildings to navigate.
# Output: 4 Files, two for describing the map used by ROS for navigation,
# and two for describing the model of the floorplan to be used by gazebo
# These files are stored in two sudirectories of kth_maps_preprocessing:
# maps and models

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

import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import math, sys, os, cv2, yaml
import numpy as np
from scipy.misc import imsave
from copy import deepcopy

res = .05

##################### parse_tree definition #################################
# parse_tree will parse the xml file defining the floorplan, using the format
# from KTH dataset. The function will return an image of the map generated as
# a numpy array in grayscal, the name of the floorplan as a string,
# the coordinates of the lower left corner of the map as a tuple (x,y) in meters,
# and the element tree used to describe the entire model of the world in a format
# useable by gazebo
#
# The file argument expects the full path to the xml floorplan description
# The res argument expects the resolution of meters/pixel in the resulting map
#############################################################################
def parse_tree(file):

    ####### Get the root of the tree, and the scale of the model #############
    tree = ET.parse(file)
    root = tree.getroot()
    name = root.get('FloorName')

    sdf = ET.Element('sdf', attrib={'version':'1.6'})
    model = ET.SubElement(sdf,'model',{'name':name})
    static = ET.SubElement(model,'static')
    static.text = '1'

    ScaleElem = root.find('Scale')

    scale = float(ScaleElem.attrib['RealDistance'])/float(ScaleElem.attrib['PixelDistance'])
    scale = 2*scale
    # Double the scale, as most doors in the floorplans are only .5 meters wide

    walls = []
    count = 0
    max_x = 0
    min_x = 999999999
    max_y = 0
    min_y = 999999999

    # Parse through each element in the xml file, a list of start and end points
    for child in root.iter('linesegment'):
        att = child.attrib
        type = att['type']

        if type == 'Wall':
            x1 = float(att['x1']) * scale
            y1 = float(att['y1']) * scale
            x2 = float(att['x2']) * scale
            y2 = float(att['y2']) * scale
            if x1 > max_x:
                max_x = x1
            if x2 > max_x:
                max_x = x2
            if x1 < min_x:
                min_x = x1
            if x2 < min_x:
                min_x = x2
            if y1 > max_y:
                max_y = y1
            if y2 > max_y:
                max_y = y2
            if y1 < min_y:
                min_y = y1
            if y2 < min_y:
                min_y = y2
            walls.append([x1,y1,x2,y2])

    walls.append([max_x,max_y,max_x,min_y])
    walls.append([max_x,max_y,min_x,max_y])
    walls.append([min_x,max_y,min_x,min_y])
    walls.append([min_x,min_y,max_x,min_y])

    # Crop the map to only include described features
    for i in range(len(walls)):
        walls[i][0] = walls[i][0] - min_x
        walls[i][1] = walls[i][1] - min_y
        walls[i][2] = walls[i][2] - min_x
        walls[i][3] = walls[i][3] - min_y

    max_x = max_x - min_x
    max_y = max_y - min_y

    width = int(max_x / res)
    height = int(max_y / res)
    thick = int(0.1 / res)
    radius = int(.7 / res)

    # Instantiate the map
    img = np.zeros((height,width),dtype=np.uint8) + 254
    inflated_map = np.zeros((height,width),dtype=np.uint8) + 254

    # For each wall, draw it on the map and add an element to the gazebo model
    for line in walls:
        x1 = int(line[0] / res)
        y1 = height - int((max_y - line[1]) / res)
        x2 = int(line[2] / res)
        y2 = height - int((max_y - line[3]) / res)
        img = cv2.line(img, (x1,y1),(x2,y2),color = 0, thickness=thick)
        linkname = 'Wall_' + str(count)
        model.append(genLink(line[0],-line[1],line[2],-line[3],linkname))
        count = count + 1

    ##### Shift the origin of the map so that a robot does not start in a wall
    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                inflated_map = cv2.circle(inflated_map, (j,i), radius, color = 0, thickness = -1)

    origin_found = False
    x = int(width / 2)
    y = int(height / 2)
    if inflated_map[y][x] != 0:
        origin_found = True
    i = 0
    while not origin_found:
        for j in [(x+i,y),(x-i,y),(x,y+i),(x,y-i)]:
            if inflated_map[j[1]][j[0]] != 0:
                origin_found = True
                x = j[0]
                y = j[1]
                break
        i = i + 1
    y = height - y
    x_pose = x * res
    y_pose = y * res

    # Include the origin of the model in the model tree
    pose = ET.SubElement(model,'pose',{'frame':''})
    pose.text = "{} {} 0 0 0 0".format(-x_pose, max_y - y_pose)
    model_tree = ET.ElementTree(sdf)

    return img, name, (-x_pose,-y_pose), model_tree

############## genLink definition #########################
# genLink is a helper function to take a start point, and end point, and a
# name for the link to generate a gazebo link description as an element tree
###########################################################
def genLink(x1,y1,x2,y2, name):
    link = ET.Element('link', {'name':name})
    # Collision
    collision = ET.SubElement(link,'collision',{'name':name + '_Collision'})
    geometry = ET.SubElement(collision,'geometry')
    box = ET.SubElement(geometry,'box')
    size = ET.SubElement(box,'size')
    length = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) )
    thick = 0.1
    height = 2.5
    size.text = "{} {} {}".format(length,thick,height)
    pose = ET.SubElement(collision,'pose',{'frame':''})
    pose.text = "{} {} {} {} {} {}".format(length/2, 0, 1.25, 0, 0, 0)

    # Visual
    visual = ET.SubElement(link,'visual',{'name':name + '_Visual'})
    pose = ET.SubElement(visual,'pose', {'frame':''})
    pose.text = "{} {} {} {} {} {}".format(length/2, 0, 1.25, 0, 0, 0)
    geometry = ET.SubElement(visual,'geometry')
    box = ET.SubElement(geometry,'box')
    size = ET.SubElement(box,'size')
    size.text = "{} {} {}".format(length,thick,height)
    material = ET.SubElement(visual,'material')
    script = ET.SubElement(material,'script')
    uri = ET.SubElement(script,'uri')
    uri.text = 'file://media/materials/scripts/gazebo.material'
    matName = ET.SubElement(script,'name')
    matName.text = 'Gazebo/Grey'
    ambient = ET.SubElement(material,'ambient')
    ambient.text = '1 1 1 1'

    # Placement and orientation
    pose = ET.SubElement(link,'pose', {'frame':''})
    yaw = math.atan2((y2 - y1),(x2 - x1))
    pose.text = "{} {} {} {} {} {}".format(x1, y1, 0, 0, 0, yaw)

    return link

def set_origin(image):
    img = deepcopy(image)
    height, width = img.shape
    radius = int(.30/res)
    ####################### Inflate the image ################################
    for i in range(width):
        for j in range(height):
            if image[j,i] == 0:
                img = cv2.circle(img,(i,j),radius,color=(0),thickness = -1)
            if image[j,i] == 254:
                img[j,i] = -1

    ######### until all white cells removed, wavefront new values ###########
    k = 1
    while -1 in img:
        start = None
        open_dict = dict()
        closed_dict = dict()
        for i in range(width):
            for j in range(height):
                if img[j,i] == -1 and start == None:
                    start = (i,j)
                open_dict[(i,j)] = False
                closed_dict[(i,j)] = False
        open_list = [start]
        while open_list:
            sample = open_list.pop(0)
        x = sample[0]
        y = sample[1]
        closed_dict[(x,y)] = True
        img[y,x] = k
        for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if open_dict[(i,j)] or i < 0 or j < 0 or i >= width or j >= height:
                pass
            else:
                if img[j,i] != -1:
                    open_dict[(i,j)] = True
                else:
                    open_dict[(i,j)] = True
                    open_list.append((i,j))
        k = k + 1

    ################## Find the biggest area ###########################
    buckets = [0] * (int(img.max()) + 1)
    for i in range(width):
        for j in range(height):
            val = img[j,i]
            buckets[val] = buckets[val] + 1

    zone = buckets.index(max(buckets[1:]))

    ########### inflate more, to avoid a starting spot next to a wall #########
    ############# find closest eligible point to the center ###################
    open_dict = dict()
    closed_dict = dict()
    for i in range(width):
        for j in range(height):
            if image[j,i] == 0:
                img = cv2.circle(img,(i,j),int(.7 / res),color=(0),thickness = -1)
            open_dict[(i,j)] = False
            closed_dict[(i,j)] = False

    start = (int(width/2),int(height/2))
    open_list = [start]
    open_dict[start] = True
    origin = None
    while not origin:
        sample = open_list.pop(0)
        x = sample[0]
        y = sample[1]
        closed_dict[sample] = True
        if img[y,x] == zone:
            origin = (x,y)
        else:
            for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                if open_dict[(i,j)] or i < 0 or j < 0 or i >= width or j >= height:
                    pass
                else:
                    open_dict[(i,j)] = True
                    open_list.append((i,j))

    ############# turn origin into metric coordinates ####################
    origin = list(origin)
    origin[1] = height - origin[1]
    origin = [-i * res for i in origin]
    return origin

###################### writeSDF ################################
# takes in the model tree for gazebo and writes out the configuration and the
# sdf file to a subdirectory of models for the floorplan
#####################################################################
def writeSDF(model_tree):
    root = model_tree.getroot()
    model = root.find('model')
    name = model.attrib['name']


    dirname = os.path.dirname(os.path.dirname(__file__))
    dirname = os.path.join(dirname,'models/'+name + '/')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    model_tree.write(dirname+'model.sdf')
    sdf = MD.parse(dirname+'model.sdf')
    sdf_pretty = sdf.toprettyxml()
    f = open(dirname+'model.sdf','w')
    f.write(sdf_pretty)
    f.close()

    config = ET.Element('model')
    conf_name = ET.SubElement(config,'name')
    conf_name.text = name
    ver = ET.SubElement(config,'version')
    ver.text = '1.0'
    conf_sdf = ET.SubElement(config,'sdf',{'version':'1.6'})
    conf_sdf.text = 'model.sdf'
    author = ET.SubElement(config, 'author')
    auth_name = ET.SubElement(author,'name')
    auth_name.text = 'Austin Whitesell'
    auth_email = ET.SubElement(author, 'email')
    auth_email.text = 'whitesea@oregonstate.edu'
    description = ET.SubElement(config,'description')
    description.text = "Gazebo model for the {} floorplan from the KTH dataset".format(name)

    config_tree = ET.ElementTree(config)
    config_tree.write(dirname+'model.config')
    conf = MD.parse(dirname+'model.config')
    conf_pretty = conf.toprettyxml()
    g = open(dirname+'model.config','w')
    g.write(conf_pretty)
    g.close()

####################### write_map ###################################
# takes in the numpy array for the image, the name of the floorplan, the
# origin in metric cartesian coordinates, and the resolution of the map
# and writes out a map.pgm and map.yaml file for ROS to use to navigate
# to a subdirectory of maps for the floorplan
def write_map(image, name, orig):
    dirname = os.path.dirname(os.path.dirname(__file__))
    dirname = os.path.join(dirname,'maps/'+ name + '/')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    imsave(dirname+"map.pgm",image)
    config = dict(
       image = "map.pgm",
       resolution = res,
       origin = [orig[0], orig[1], 0.000],
       negate = 0,
       occupied_thresh = 0.65,
       free_thresh = 0.196
    )
    with open(dirname + "map.yaml", "w") as outfile:
        yaml.dump(config,outfile,default_flow_style=False)

def main(args):
    try:
        img, name, origin, model_tree = parse_tree(args[1])
        origin = set_origin(img)
        write_map(img,name, origin)
        writeSDF(model_tree)
    except:
        print args[1]
    #     # exit(1)
if __name__ == "__main__":
    main(sys.argv)
