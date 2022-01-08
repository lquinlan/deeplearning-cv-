from Filtering import Filter_sp as sp
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img=cv.imread('4.jpg',0)
filter=sp(img)
img1=filter.gauss_filter(0.3)
img2=filter.gauss_filter(0.4)
img3=img2-img1
img1=filter.gauss_filter(0.6)
img2=filter.gauss_filter(0.7)
img4=img2-img1
img1=filter.gauss_filter(0.7)
img2=filter.gauss_filter(0.8)
img5=img2-img1
plt.subplot(221)
plt.imshow(img3,cmap='gray')
plt.subplot(222)
plt.imshow(img4,cmap='gray')
plt.subplot(223)
plt.imshow(img5,cmap='gray')
plt.show()
mat=[]
for i in range(1,img4.shape[0]-1):
    for j in range(1,img4.shape[1]-1):
        tmp=np.abs(img4[i][j])
        t_max=max(np.max(img3[i-1:i+2,j-1:j+2]),np.max(img4[i-1:i+2,j-1:j+2]),np.max(img5[i-1:i+2,j-1:j+2]))
        t_min=min(np.min(img3[i-1:i+2,j-1:j+2]),np.min(img4[i-1:i+2,j-1:j+2]),np.min(img5[i-1:i+2,j-1:j+2]))
        # print(t_max,t_min)
        f_max=np.abs([t_max,t_min])
        if tmp==t_max and tmp>=30:
            mat.append([i,j])
            
mat=np.array(mat)
print(mat.shape)
plt.imshow(img,cmap='gray')
plt.scatter(mat[:,1],mat[:,0],marker='x')
plt.show()