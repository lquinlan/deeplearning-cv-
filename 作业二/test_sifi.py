import cv2 as cv
img=cv.imread('2.jpg',0)
newimg=img
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(newimg,None)
print(kp[0].pt)
print(len(kp))
print(des)
print(des.shape)
points2f = cv.KeyPoint_convert(kp) 
print(points2f)
img=cv.drawKeypoints(newimg,kp,img)
cv.imshow('kp',img)

cv.waitKey()