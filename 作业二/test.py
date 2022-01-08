# import numpy as np
# from skimage import filters
# from Filtering import Filter_sp as filter
# import matplotlib.pyplot as plt
# import cv2 as cv
# from skimage.feature import corner_peaks
# # def gaussian_kernel(size,sigma):
# #     gaussian_kernel=np.zeros((size,size))
# #     for i in range(size):
# #         for j in range(size):
# #             x = i - (size-1)/2
# #             y = j - (size-1)/2
# #             gaussian_kernel[i,j]=(1/(2*np.pi*sigma**2))*np.exp(-(x**2 + y**2) / (2*sigma**2))
# #     return gaussian_kernel

# # def conv(image,kernel):
# #     m,n = image.shape
# #     kernel_m,kernel_n = kernel.shape
# #     image_pad = np.pad(image,((kernel_m//2,kernel_m//2),(kernel_n//2 , kernel_n//2)),'constant')
# #     result = np.zeros((m,n))
# #     for i in range(m):
# #         for j in range(n):
# #             value = np.sum(image_pad[i:i+kernel_m,j:j+kernel_n]*kernel)
# #             result[i,j]=value
# #     return result

# # def harris_corners(image,window_size=3,k=0.04,window_type=0):
# #     if window_type==0:
# #         window=np.ones((window_size,window_size))
# #     if window_type==1:
# #         window = gaussian_kernel(window_size,1)
# #     m,n = image.shape
# #     dx = filters.sobel_v(image)
# #     dy = filters.sobel_h(image)
# #     dx_dx = dx * dx
# #     dy_dy = dy * dy
# #     dx_dy = dx * dy
# #     plt.imshow(dx,cmap='gray')
# #     plt.show()
# #     plt.imshow(dy,cmap='gray')
# #     plt.show()
# #     w_dx_dx = conv(dx_dx,window)
# #     w_dy_dy = conv(dy_dy,window)
# #     w_dx_dy = conv(dx_dy,window)
# #     reponse = np.zeros((m,n))
# #     for i in range(m):
# #         for j in range(n):
# #             M=np.array([[w_dx_dx[i,j],w_dx_dy[i,j]],[w_dx_dy[i,j],w_dy_dy[i,j]]])
# #             R = np.linalg.det(M)-k*(np.trace(M))**2
# #             reponse[i,j] = R
# #     return reponse
# # plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
# # plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'


# from skimage.util import view_as_blocks
# # patch:所要进行hog描述的区域
# # cell_size:对所要描述的区域进行划分，每个区域有8*8，64个像素，将这64个像素的幅值，映射到，180/2的九个直方图中
# def hog_description(patch,cell_size=(8,8)):
    
#     if patch.shape[0] % cell_size[0]!=0 or patch.shape[1] % cell_size[1]!=0:
#         return 'The size of patch and cell don\'t match'
#     # n_bins=9直方图大小
#     n_bins=9
#     # 180/20=9
#     degree_per_bins=20
#     # 计算x方向和y方向的梯度
#     Gx = filter(patch,(3,1)).garx1()
#     Gy = filter(patch,(1,3)).gary1()
#     # Gx = filters.sobel_v(patch)
#     # Gy = filters.sobel_h(patch)
#     plt.imshow(Gx)
#     plt.show()
#     # 计算梯度的大小
#     G = np.sqrt(Gx**2 + Gy**2)
#     # 计算梯度的方向，并将其映射到0-180
#     theta = (np.arctan2(Gy,Gx) * 180 / np.pi) % 180
#     # 对描述大小和方向的描述图进行cell划分
#     G_as_cells = view_as_blocks(G,block_shape=cell_size)
#     theta_as_cells = view_as_blocks(theta,block_shape=cell_size)
#     # h和w分别是划分后图像的高和宽分别变为多少
#     H = G_as_cells.shape[0]
#     W = G_as_cells.shape[1]
#     # bin_accumulator ：hog描述的存储矩阵
#     bins_accumulator = np.zeros((H,W,n_bins))
#     for i in range(H):
#         for j in range(W):
#             # 对每一个划分进行hoG计算
#             theta_cell = theta_as_cells[i,j,:,:]
#             G_cell = G_as_cells[i,j,:,:]
            
#             for p in range(theta_cell.shape[0]):
#                 for q in range(theta_cell.shape[1]):
#                     theta_value = theta_cell[p,q]
#                     G_value = G_cell[p,q]
#                     # 获得分配到第几个直方图中，但未取余
#                     num_bins = int(theta_value // degree_per_bins)
#                     # 算距离其最近直方图的距离
#                     k= int(theta_value % degree_per_bins)
#                     # 按距离比列进行分配
#                     bins_accumulator[i,j,num_bins % n_bins] += (degree_per_bins - k) / degree_per_bins* G_value
#                     bins_accumulator[i,j,(num_bins+1) % n_bins] += k / degree_per_bins * G_value
#             # print(bins_accumulator[i][j])
#             # plt.bar(range(9),bins_accumulator[i][j])
#             # plt.show()
 
#     Hog_list = []
#     for x in range(H-1):
#         for y in range(W-1):
#             block_description = bins_accumulator[x:x+2,y:y+2]
#             block_description = block_description / np.sqrt(np.sum(block_description**2))
#             Hog_list.append(block_description)
#             for a in range(2):
#                 for b in range (2):
#                     print(block_description[a][b])
#                     plt.bar(range(9),block_description[a][b])
#                     plt.show()
#     # 返回时将其展开成一维的
#     # 7*15*9*4=3780
#     return np.array(Hog_list).flatten()
# #Harris Corners Detector
# image = cv.imread('2.jpg',0)
# image=image[0:128,0:256]
# reponse = hog_description(image)
# # print(reponse.shape)
# # plt.subplot(211)
# plt.imshow(image)

# # plt.subplot(212)
# # plt.imshow(reponse)
# plt.show()

# def keypoint_description(image,keypoint,desc_func,patch_size=16):
#     keypoint_desc = []
#     for i,point in enumerate(keypoint):
#         x,y = point
#         patch = image[x-patch_size//2:x+int(np.ceil(patch_size/2)),y-patch_size//2:y+int(np.ceil(patch_size/2))]
#         description = desc_func(patch)
#         keypoint_desc.append(description)
#     return np.array(keypoint_desc)
# from scipy.spatial.distance import cdist
# def description_matches(desc1,desc2,threshold=0.5):
#     distance_array = cdist(desc1,desc2)
#     matches = []
#     i=0
#     for each_distance_list in distance_array:
#         arg_list = np.argsort(each_distance_list)
#         index1 = arg_list[0]
#         index2 = arg_list[1]
#         if each_distance_list[index1] / each_distance_list[index2] <= threshold:
#             matches.append([i,index1])
#         i+=1
#     return np.array(matches)
import numpy as np
a=np.array([[1,2,2],[2,2,2]])
b=np.array([[3,4,5],[6,7,8]])
c=[a,b]
c=np.vstack(c)
print(c.shape)
print(c)
print(np.max(c,axis=0))
print(np.eye(3))
print(np.argmax(c,axis=1))
# 求图形的仿射变换
def fit_affine_matrix(p1,p2):
    assert (p1.shape[0]==p2.shape[0]),'The number of p1 and p2 are different'
    # 加上偏置
    p1=np.hstack((p1,np.ones((p1.shape[0],1))))
    p2=np.hstack((p2,np.ones((p2.shape[0],1))))
    # ap2=p1
    H = np.linalg.pinv(p2) @ p1
    # 因为仿射变换矩阵的性质
    H[:,2]=np.array([0,0,1])
    return H

def ransac(keypoint1,keypoint2,matches,n_iters=200,threshold=20):
    N=matches.shape[0]
    match_keypoints1 = np.hstack((keypoint1[matches[:,0]],np.ones((N,1))))
    match_keypoints2 = np.hstack((keypoint2[matches[:,1]],np.ones((N,1))))
    n_samples=int(N*0.2)
    n_max = 0
    for i in range(n_iters):
        random_index = np.random.choice(N,n_samples,replace=False)
        p1_choice = match_keypoints1[random_index]
        p2_choice = match_keypoints2[random_index]
        H_choice = np.linalg.pinv(p2_choice) @ p1_choice
        H_choice[:,2] = np.array([0,0,1])
        p1_test = match_keypoints2 @ H_choice
        diff = np.sum((match_keypoints1[:,:2]-p1_test[:,:2])**2,axis=1)
        index=np.where(diff<=threshold)[0]
        n_index = index.shape[0]
        if n_index>n_max:
            H=H_choice
            robust_matches=matches[index]
            n_max=n_index
    return H,robust_matches