import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import math, sys, os

def model(file):
    tree = ET.parse(file)
    root = tree.getroot()
    name = root.get('FloorName')

    sdf = ET.Element('sdf', attrib={'version':'1.6'})
    model = ET.SubElement(sdf,'model',{'name':name})
    static = ET.SubElement(model,'static')
    static.text = '1'

    ScaleElem = root.find('Scale')

    scale = float(ScaleElem.attrib['RealDistance'])/float(ScaleElem.attrib['PixelDistance'])

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
            y1 = -float(att['y1']) * scale
            x2 = float(att['x2']) * scale
            y2 = -float(att['y2']) * scale
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
            linkname = 'Wall_' + str(count)
            model.append(genLink(x1,y1,x2,y2,linkname))
            count = count + 1

    linkname = 'Wall_' + str(count)
    model.append(genLink(min_x,min_y,min_x,max_y, linkname))
    count = count + 1
    linkname = 'Wall_' + str(count)
    model.append(genLink(min_x,max_y,max_x,max_y, linkname))
    count = count + 1
    linkname = 'Wall_' + str(count)
    model.append(genLink(min_x,min_y,max_x,min_y, linkname))
    count = count + 1
    linkname = 'Wall_' + str(count)
    model.append(genLink(max_x,min_y,max_x,max_y, linkname))
    x_pose = (max_x - min_x)/2 + min_x
    y_pose = (max_y - min_y)/2 + min_y
    x_pose = -1*x_pose
    y_pose = -1*y_pose
    pose = ET.SubElement(model,'pose',{'frame':''})
    pose.text = "{} {} 0 0 0 0".format(x_pose,y_pose)
    model_tree = ET.ElementTree(sdf)
    return model_tree

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

def main(args):
    try:
        model_tree = model(args[1])
        writeSDF(model_tree)
    except:
        print args[1]
        exit(1)
if __name__ == "__main__":
    main(sys.argv)
