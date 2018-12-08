import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import math, sys, os, cv2, yaml
from copy import deepcopy
import numpy as np
from scipy.misc import imsave

res = .05

def parse_tree(file):
    tree = ET.parse(file)
    root = tree.getroot()
    name = root.get('FloorName')

    ScaleElem = root.find('Scale')

    scale = float(ScaleElem.attrib['RealDistance'])/float(ScaleElem.attrib['PixelDistance'])
    scale = 2*scale

    walls = []
    count = 0
    max_x = 0
    min_x = 999999999
    max_y = 0
    min_y = 999999999
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
            walls.append((x1,y1,x2,y2))

    walls.append((max_x,max_y,max_x,min_y))
    walls.append((max_x,max_y,min_x,max_y))
    walls.append((min_x,max_y,min_x,min_y))
    walls.append((min_x,min_y,max_x,min_y))


    x_pose = (max_x - min_x)/2
    y_pose = (max_y - min_y)/2
    x_pose = -1*x_pose
    y_pose = -1*y_pose
    width = int((max_x - min_x) / res)
    height = int((max_y - min_y) / res)
    thick = int(0.1 / res)

    img = np.zeros((height,width),dtype=np.uint8) + 254
    for line in walls:
        x1 = int((line[0] - min_x) / res)
        y1 = height - int((max_y - line[1]) / res)
        x2 = int((line[2] - min_x) / res)
        y2 = height - int((max_y - line[3]) / res)
        img = cv2.line(img,(x1,y1),(x2,y2),color = 0, thickness=thick)

    return img, name, (x_pose,y_pose)

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

    height, width = image.shape
    origin_image = deepcopy(image)
    origin_image = cv2.cvtColor(origin_image,cv2.COLOR_GRAY2RGB)
    x_org = -int(orig[0] / res)
    y_org = height + int(orig[1] / res)
    origin_image = cv2.circle(origin_image,(x_org,y_org),int(.3/res),color=(0,255,0), thickness = -1)
    for i in range(width):
        for j in range(height):
            if image[j,i] == 0:
                origin_image = cv2.circle(origin_image,(i,j),int(.3 / res),color=(0,0,0),thickness = -1)
    cv2.imwrite(dirname + "origin.png", origin_image)


def main(args):
    try:
        img, name, origin = parse_tree(args[1])
        origin = set_origin(img)
        write_map(img,name, origin)
    except:
        print args[1]
    #     # exit(1)
if __name__ == "__main__":
    main(sys.argv)
