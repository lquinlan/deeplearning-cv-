import cv2 as cv
from numpy.core.fromnumeric import size
from Filtering import Filter_sp 
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_peaks
def harris (img,wsize=(5,5),window_type=0,threshold=0.01,k=0.04):
    img_nwe=np.copy(img)
    # w=1.03  1效果还可以
    filter1=Filter_sp(img_nwe,size=(5,5),w=1.03)
    
    # img_nwe=filter1.gauss_filter(sigma=0.08)
    img_nwe=filter1.high_filter()
    filter2=Filter_sp(img_nwe,(3,1))
    filter3=Filter_sp(img_nwe,(1,3))
    Ix=filter2.garx1()
    Iy=filter3.gary1()
    # print(Ix)

    Ix2=Ix*Ix
    # print(Ix2)
    Iy2=Iy*Iy
    Ixy=Ix*Iy
    # M=np.array([[Ix2,Ixy],[Ixy,Iy2]])
    if window_type==0:
        Ix2=Filter_sp(Ix2,size=wsize).aver_filter()*25
        Iy2=Filter_sp(Iy2,size=wsize).aver_filter()*25
        Ixy=Filter_sp(Ixy,size=wsize).aver_filter()*25
    else:
        Ix2=Filter_sp(Ix2,size=wsize).gauss_filter(0.8)
        Iy2=Filter_sp(Iy2,size=wsize).gauss_filter(0.8)
        Ixy=Filter_sp(Ixy,size=wsize).gauss_filter(0.8)
    harris_des=np.zeros(img.shape)
    # print(Ix2)

    for i in range(20,img.shape[0]-20):
        for j in range(20,img.shape[1]-20):
            m=np.array([[Ix2[i][j],Ixy[i][j]],[Ixy[i][j],Iy2[i][j]]])
            tmp=np.linalg.det(m)-k*(np.trace(m))**2
            harris_des[i][j]=tmp
    ha=corner_peaks(harris_des,threshold_rel=threshold)
    return ha
