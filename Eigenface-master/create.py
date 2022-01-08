import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
IMAGE_SIZE =(50,50)
name=set()
def createDatabase(path):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)
    # print(TrainFiles)
    label=[]
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    Train_Number = len(TrainFiles)
    T = []
    # 把所有图片转为1-D并存入T中
    for i in range(0,Train_Number):
        name.add(str(TrainFiles[i][0:5]))
    print(len(list(name)))
    # print(T.shape)
    # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    
createDatabase('./jaffe')
 