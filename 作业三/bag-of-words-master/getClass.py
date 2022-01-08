import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *
import matplotlib.pyplot as plt 
# Load the classifier, class names, scaler, number of clusters and vocabulary 

fn='150bof.pkl'
clf, classes_names, k, voc = joblib.load(fn)
true_lab=[]

path='./dataset/test'
# Get the path of the testing image(s) and store them in a list
image_paths = []
id=0
testing_names = os.listdir(path)
for testing_name in testing_names:
        dir = os.path.join(path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
        true_lab+=[id]*len(class_path)
        id+=1
        

# Create feature extraction and keypoint detector objects
fea_det = cv2.SIFT_create()
des_ext = fea_det

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor)) 

# 
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    if des_list[i][1] is not None:
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            test_features[i][w] += 1


# test_features=joblib.load('test.out')

predictions =  [classes_names[i] for i in clf.predict(test_features)]
print(clf.score(test_features,true_lab))
for i ,pre in enumerate(predictions):
    im = cv2.imread(image_paths[i])
    plt.subplot(3,7,i+1)
    
    plt.title(pre)
    plt.axis("off")
    if len(im.shape) == 2:
        plt.imshow(im, cmap = "gray")
    else:
        im_display = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        plt.imshow(im_display)
plt.show()

