import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

path='./dataset/train'
training_names = os.listdir(path)
print(training_names)
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

fea_det = cv2.SIFT_create()
des_ext = fea_det
des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))   
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor))  

joblib.dump(descriptors,'a.out')
print('合并完成')

k = 120

from sklearn.svm import SVC
voc, variance = kmeans(descriptors, k) 

print('聚类完成')

im_features = np.zeros((len(image_paths), k), "float32")
for i in range (len(image_paths)):
    if des_list[i][1] is not None:
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
print('词袋')
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))
# svc=SVC(max_iter=10000)
# svc.fit(im_features, np.array(image_classes))
print('svm')
# Save the SVM
fname=str(k)+'bof.pkl'
joblib.dump((clf, training_names, k, voc), fname, compress=3)    
# joblib.dump((svc, training_names, k, voc), 'scvbof.pkl', compress=3)    
