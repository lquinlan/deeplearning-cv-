import os
import csv
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt

from mindspore import nn, context
from mindspore.ops import operations as ops
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class Pca(nn.Cell):

    def __init__(self):
        super(Pca, self).__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reshape = ops.Reshape()
        # self.matmul_a = ops.MatMul()
        self.matmul_a = ops.MatMul(transpose_a=True)
        self.top_k = ops.TopK(sorted=True)
        self.gather = ops.GatherV2()

    def construct(self, x, dim=2):
        '''
        x:输入矩阵
        dim:降维之后的维度数
        '''
        
        m = x.shape[0]
        # 计算张量的各个维度上的元素的平均值
        mean = self.reduce_mean(x, axis=0)
        # print(mean)
        # 去中心化
        x_new = x - mean
        x_new=x_new.asnumpy()
        # 无偏差的协方差矩阵
        # print('rr')
        cov = x_new.T@x_new
        # print('dff')
        # print(cov) 
        # 计算特征分解
        # cov = cov.asnumpy()
        e, v = np.linalg.eigh(cov)
        # 将特征值从大到小排序，选出前dim个的index
        
        # 提取前排序后dim个特征向量
        pca = []
        
        z=np.argsort(-e) #按降序排列
        for i in range(0,dim):
        # print(V[z[i]])
            pca.append(v[:,z[i]])
        return np.array(pca)



