#!/usr/bin/env python
import os
if not 'DISPLAY' in os.environ.keys():
    import matplotlib as mpl
    mpl.use('Agg')
from keras.models import load_model
from glob import glob
import numpy as np
from scipy.misc import imread
import yaml, tqdm, time, sys
import matplotlib.pyplot as plt

np.random.seed(int(time.time()))

def generate_curve_from_files(prediction_file,label_file):
    predictions = np.load(prediction_file)
    labels = np.load(label_file)
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
    return((TPR,FPR))

class model_tester:
    def __init__(self,model_file):
        self.model = load_model(model_file)
        self.predictions = []
        self.labels = []
        self.predictions = []

    def generate_curves(self,model_folder,window_size):
        folders = self._get_datafolders(model_folder)
        self._get_predictions(folders,window_size)
        TPR = []
        FPR = []
        for i in np.linspace(0,1,100):
            generated_labels = [j >= i for j in self.predictions]
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
            TPR.append(float(TP) / float(TP + FN))
            FPR.append(float(FP) / float(FP + TN))
        return((TPR,FPR))

    def save_predictions(self,filename):
        np.save(filename,self.predictions)
    def save_labels(self,filename):
        np.save(filename,self.labels)

    def _get_predictions(self,folders,window_size):
        for folder in tqdm.tqdm(folders):
            data = self._data_set_from_folder(folder,window_size)
            input = []
            labels = []
            for point in data:
                input.append(list(point[0]/255))
                labels.append(point[1])
            input = np.array(input)
            input = input.reshape(input.shape[0],input.shape[1],input.shape[2],1)
            predictions = self.model.predict_proba(input)
            self.labels = self.labels + labels
            self.predictions = self.predictions + list(predictions[:,0])


    def _data_set_from_folder(self,folder_path, window_size):
        im  = imread(folder_path + "map.pgm")
        height,width = im.shape
        config = yaml.load(open(folder_path+'map.yaml','r'))
        res = config['resolution']
        origin = config['origin']
        x_org = -origin[0]
        y_org = -origin[1]

        datafiles = glob(folder_path + "/*.npy")
        labels = np.zeros((height,width))

        horizontal_edge = np.ones((int(5/res),width)) * 205
        vertical_edge = np.ones((height+(int(10/res)),int(5/res))) * 205
        im = np.concatenate((im,horizontal_edge),axis=0)
        im = np.concatenate((horizontal_edge,im),axis=0)
        im = np.concatenate((vertical_edge,im),axis=1)
        im = np.concatenate((im,vertical_edge),axis=1)
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
        extended_im = self._extend_image(im,window_size,res)
        for i in range(num):
            ind = np.random.choice(len(failure_locations))
            fail_y,fail_x = failure_locations[ind]
            del failure_locations[ind]
            output.append([self._get_window(extended_im,fail_y,fail_x,window_size,res),labels[y,x] - 1])
            ind = np.random.choice(len(success_locations))
            y,x = success_locations[ind]
            del success_locations[ind]
            output.append([self._get_window(extended_im,y,x,window_size,res),labels[y,x] - 1])
        # for x in range(width):
        #     for y in range(height):
        #         if labels[y,x]:
        #             output.append([self._get_window(im,res,y,x),labels[y,x] - 1])
        return np.array(output)

    def _get_window(self,im,y,x,window_size,res):
        left_bound = x
        right_bound = x + int(float(window_size)/float(res))
        upper_bound = y
        lower_bound = y + int(float(window_size)/float(res))
        return im[upper_bound:lower_bound,left_bound:right_bound]

    def _extend_image(self,im,window_size,res):
        height,width = im.shape
        horizontal_edge = np.ones((int((window_size*.5)/res),width)) * 205
        vertical_edge = np.ones((height+(int(window_size/res)),int((window_size*.5)/res))) * 205
        new_im = np.concatenate((im,horizontal_edge),axis=0)
        new_im = np.concatenate((horizontal_edge,new_im),axis=0)
        new_im = np.concatenate((vertical_edge,new_im),axis=1)
        new_im = np.concatenate((new_im,vertical_edge),axis=1)
        return new_im

    def _get_datafolders(self,model_folder):
        folders = []
        # This is bad, but I don't want to make this super robust right now
        package_folder = "/home/whitesea/workspace/data_post_processing/"
        with open(package_folder + model_folder + 'training_validation_split.txt') as f:
            flag = False
            for line in f.readlines():

                if flag:
                    chunks = line.split('/')
                    world = chunks[1]
                    folders.append(package_folder + 'data/' + world + '_ros/')
                elif 'validation files' in line:
                    flag = True
        print(folders)
        return folders

if __name__ == "__main__":

    model_folder = sys.argv[1]
    window_size = float(sys.argv[2])
    tester = model_tester(model_folder+'failure_model.h5')
    TPR,FPR = tester.generate_curves(model_folder,window_size)
    tester.save_labels(model_folder + 'labels.npy')
    tester.save_predictions(model_folder+ 'predictions.npy')
    plt.plot(FPR,TPR, '-o')
    plt.plot([0,1],[0,1],'--r')
    plt.title('ROC curve of failure_model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(model_folder + 'roc_curve.png')

    # TPR,FPR = generate_curve_from_files(model_folder+'predictions.npy',model_folder+'labels.npy')
    # plt.plot(FPR,TPR,'-o')
    # plt.plot([0,1],[0,1],'--r')
    # plt.title('ROC curve of failure_model')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.savefig(model_folder + 'roc_curve.png')
