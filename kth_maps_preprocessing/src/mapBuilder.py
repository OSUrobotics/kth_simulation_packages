import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import math, sys, os, cv2, yaml
import numpy as np
from scipy.misc import imsave


def parse_tree(file, res):
    tree = ET.parse(file)
    root = tree.getroot()
    name = root.get('FloorName')

    ScaleElem = root.find('Scale')

    scale = float(ScaleElem.attrib['RealDistance'])/float(ScaleElem.attrib['PixelDistance'])

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

def write_map(image, name, orig, res):
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
    res = .05
    try:
        img, name, origin = parse_tree(args[1],res)
        write_map(img,name, origin, res)
    except:
        print args[1]
        # exit(1)
if __name__ == "__main__":
    main(sys.argv)
