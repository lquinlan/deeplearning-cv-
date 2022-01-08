from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import haaris
import hog
import numpy as np
import cv2 as cv
import time
def key_des(img,key,desc_func,patch_size=16):
    keypoint_desc = []
    for i,point in enumerate(key):
        x,y = point
        patch = img[x-patch_size//2:x+int(np.ceil(patch_size/2)),y-patch_size//2:y+int(np.ceil(patch_size/2))]
        # print(patch.shape)
        description = desc_func(patch)
        # if description=='0':
        #     print(x,y)
        #     print(patch.shape)
        keypoint_desc.append(description)
    return np.array(keypoint_desc)

def description_matches(desc1,desc2,threshold=0.5):
    distance_array = cdist(desc1,desc2)
    matches = []
    i=0
    for each_distance_list in distance_array:
        arg_list = np.argsort(each_distance_list)
        index1 = arg_list[0]
        index2 = arg_list[1]
        if each_distance_list[index1] / each_distance_list[index2] <= threshold:
            matches.append([i,index1])
        i+=1
    return np.array(matches)
def plot_matches(ax,image1,image2,keypoint1,keypoint2,matches):
    H1,W1 = image1.shape
    H2,W2 = image2.shape
    if H1>H2:
        new_image2 = np.zeros((H1,W2))
        new_image2[:H2,:] = image2
        image2 = new_image2
    if H1<H2:
        new_image1 = np.zeros((H2,W1))
        new_image1[:H1,:]=image1
        image1 = new_image1
    image = np.concatenate((image1,image2),axis=1)
    ax.scatter(keypoint1[:,1],keypoint1[:,0],facecolors='none',edgecolors='k')
    ax.scatter(keypoint2[:,1]+image1.shape[1],keypoint2[:,0],facecolors='none',edgecolors='k')
    ax.imshow(image,interpolation='nearest',cmap='gray')
    for one_match in matches:
        index1 = one_match[0]
        index2 = one_match[1]
        color = np.random.rand(3)
        ax.plot((keypoint1[index1,1],keypoint2[index2,1] + image1.shape[1]),
                (keypoint1[index1,0],keypoint2[index2,0]),'-',color=color)
def ransac(keypoint1,keypoint2,matches,n_iters=200,threshold=20):
    N=matches.shape[0]
    # 横向合并，类似于ax+b=0变换为ax=0给系数矩阵加一列1变为增广矩阵
    match_keypoints1 = np.hstack((keypoint1[matches[:,0]],np.ones((N,1))))
    match_keypoints2 = np.hstack((keypoint2[matches[:,1]],np.ones((N,1))))
    n_samples=int(N*0.2)
    n_max = 0
    H=np.zeros((3,3))
    robust_matches=matches
    for i in range(n_iters):
        # 生成0-100的20个下标，不重复
        random_index = np.random.choice(N,n_samples,replace=False)
        p1_choice = match_keypoints1[random_index]
        p2_choice = match_keypoints2[random_index]
        # print('shapeff')
        # print(p1_choice.shape)
        # print(p2_choice.shape)
        # w.Tp2=p1
        '''
          [[a,b,c],[d,e,f],[0,0,1]].T
        '''
        H_choice = np.linalg.pinv(p2_choice) @ p1_choice
        # print(H_choice)
        H_choice[:,2] = np.array([0,0,1])
        # print(H_choice)
        # print('shape:')
        # print('H_choice:',H_choice.shape)
        p1_test = match_keypoints2 @ H_choice
        diff = np.sum((match_keypoints1[:,:2]-p1_test[:,:2])**2,axis=1)
        # print('diff:')
        # print(diff.shape)
        index=np.where(diff<=threshold)[0]
        # print('where:')
        # print(np.where(diff<=threshold))
        n_index = index.shape[0]
        if n_index>n_max:
            H=H_choice
            robust_matches=matches[index]
            n_max=n_index
    return H,robust_matches
# 大体上根据参考矩阵和放射变换，将其他图像转换到参考图像
def get_output_space(image_ref,images,transforms):
    H_ref , W_ref = image_ref.shape
    corner_ref = np.array([[0,0,1],[H_ref,0,1],[0,W_ref,1],[H_ref,W_ref,1]])
    
    # print('shape')
    # print(corner_ref)
    all_corners=[corner_ref]
    if len(images) != len(transforms):
        print('The size of images and transforms does\'t match')
    for i in range(len(images)):
        H,W = images[i].shape
        corner = np.array([[0,0,1],[H,0,1],[0,W,1],[H,W,1]]) @ transforms[i]
        all_corners.append(corner)
    all_corners = np.vstack(all_corners)
    # print(all_corners.shape)
    max_corner = np.max(all_corners,axis=0)
    # print(max_corner)
    min_corner = np.min(all_corners,axis=0)
    # print(min_corner)
    out_space = np.ceil((max_corner - min_corner)[:2]).astype(int)
    # print(out_space)
    offset = min_corner[:2]
    # print(offset)
    return out_space,offset
        

def warp_image(image, H, output_shape, offset):
     H_invT = np.linalg.inv(H.T)
     matrix = H_invT[:2,:2]
     o = offset+H_invT[:2,2]
     image_warped = affine_transform(image,matrix,o,output_shape,cval=-1)
     return image_warped
def linear_blend(image1_warped,image2_warped):
    merged = image1_warped + image2_warped
    H , W = image1_warped.shape
    image1_mask = (image1_warped!=0)
    image2_mask = (image2_warped!=0)
    left_margin = np.argmax(image2_mask,axis=1)
    right_margin = W - np.argmax(np.fliplr(image1_mask),axis=1)
    for i in range(H):
        k = right_margin[i] - left_margin[i]
        for j in range(k):
            alpha = j / (k - 1)
            # alpha=0.5
            merged[i,left_margin[i]+j] = (1-alpha) * image1_warped[i,left_margin[i]+j]+\
            alpha * image2_warped[i,left_margin[i]+j]
    return merged
img1=cv.imread('0102.jpg',0)
img2=cv.imread('0304.jpg',0)
s=time.perf_counter()
plt.subplot(121)
plt.imshow(img1,cmap='gray')
# plt.scatter(img1key1[:,1],img1key1[:,0],marker='o')
plt.subplot(122)
plt.imshow(img2,cmap='gray')
# plt.scatter(img2key2[:,1],img2key2[:,0],marker='o')
plt.show()
img1key1=haaris.harris(img1)
img2key2=haaris.harris(img2)
# print(img1key1.shape)
# print(img2key2.shape)
plt.subplot(121)
plt.imshow(img1,cmap='gray')
plt.scatter(img1key1[:,1],img1key1[:,0],marker='o')
plt.subplot(122)
plt.imshow(img2,cmap='gray')
plt.scatter(img2key2[:,1],img2key2[:,0],marker='o')
plt.show()
des2=key_des(img2,img2key2,hog.hog_des)
des1=key_des(img1,img1key1,hog.hog_des)

# print('df')
# print(des1.shape)
# print(des2.shape)
hog_m=description_matches(des1,des2,threshold=0.7)
H,hog_m=ransac(img1key1,img2key2,hog_m)

e=time.perf_counter()
print(e-s)
fig,ax = plt.subplots(1,1)
ax.axis('off')
plot_matches(ax,img1,img2,img1key1,img2key2,hog_m)

plt.show()
output_shape, offset = get_output_space(img1, [img2], [H])

image1_warped = warp_image(img1,np.eye(3),output_shape,offset)
image1_mask = (image1_warped != -1)
image1_warped[~image1_mask]=0

image2_warped = warp_image(img2,H,output_shape,offset)
image2_mask = (image2_warped != -1)
image2_warped[~image2_mask]=0

plt.figure()
plt.subplot(121)
plt.imshow(image1_warped,cmap='gray')
plt.subplot(122)
plt.imshow(image2_warped,cmap='gray')
plt.show()

merged = image1_warped + image2_warped

overlap = np.maximum(image1_mask*1+image2_mask,1)
merged = merged / overlap

plt.figure()
plt.imshow(merged,cmap='gray')
plt.show()

merged = linear_blend(image1_warped,image2_warped)
plt.figure()
plt.imshow(merged,cmap='gray')
# plt.savefig('0304.jpg')
cv.imwrite('1234.jpg',merged)
plt.show()
