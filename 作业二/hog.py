from skimage.util import view_as_blocks
import numpy as np
from Filtering import Filter_sp as filter
def hog_des (img,cell=(8,8)):
    if img.shape[0]%cell[0]!=0 or img.shape[1]%cell[1]!=0:
        print('尺寸不匹配！')
        return '0'
    n_bins=9
    degree_=20
    gx=filter(img,(3,1)).garx1()
    gy=filter(img,(1,3)).gary1()
    gralen=np.sqrt(gx**2+gy**2)
    gradegree=(np.arctan2(gy,gx)*180/np.pi)%180
    gralen_cell=view_as_blocks(gralen,block_shape=cell)
    gradegree_cell=view_as_blocks(gradegree,block_shape=cell)
    H_after_cell=gradegree_cell.shape[0]
    W_after_cell=gradegree_cell.shape[1]
    # print('ff')
    # print(H_after_cell)
    # 
    # print(W_after_cell)
    # print(gralen_cell.shape)
    bin_set=np.zeros((H_after_cell,W_after_cell,n_bins))
    for i in range(0,H_after_cell):
        for j in range(0,W_after_cell):
            tmp_glen_cell=gralen_cell[i,j,:,:]
            tmp_gdeg_cell=gradegree_cell[i,j,:,:]
            for i_ in range(0,cell[0]):
                for j_ in range(0,cell[1]):
                    sele_bin1=int((tmp_gdeg_cell[i_][j_]//degree_)%n_bins)
                    sele_bin2=int((tmp_gdeg_cell[i_][j_]//degree_+1)%n_bins)
                    dis=tmp_gdeg_cell[i_][j_]%degree_
                    # print(sele_bin1)
                    # print(sele_bin2)
                    # print(dis)
                    bin_set[i,j,sele_bin1]+=(1-dis/degree_)*tmp_glen_cell[i_][j_]
                    bin_set[i][j][sele_bin2]+=(dis/degree_)*tmp_glen_cell[i_][j_]
            
    hog_list=[]
    for i in range(0,H_after_cell-1):
        for j in range(0,W_after_cell-1):
            hogstd=bin_set[i:i+2,j:j+2]
            hogstd=hogstd/np.sqrt(np.sum(hogstd**2))
            hog_list.append(hogstd)
            # for a in range(2):
            #     for b in range (2):
            #         print(hogstd[a][b])
            #         plt.bar(range(9),hogstd[a][b])
            #         plt.show()
                 
    return np.array(hog_list).flatten()
