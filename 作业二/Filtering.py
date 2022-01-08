import numpy as np

import math

# axis 0,1,2,3垂直，水平，主队角，副对角
class Filter_sp():
    def __init__(self,img,size=(3,3),str=1,padding='same',w=1,axis=0) -> None:
        self.img=img
        # self.H=img.shape[0]
        # self.W=img.shape[1]
        self.kernal_shape=size
        self.padding=padding
        self.str=str
        self.high_w=w
        self.axis=axis
        if self.padding=='same':
            self.tr_shape=np.array(self.img.shape)
        else:
            trx=(self.img.shape[0]-self.kernal_shape[0])//str+1
            trY=(self.img.shape[1]-self.kernal_shape[1])//str+1
            self.tr_shape=np.array([trx,trY])
        self.filte_img=np.zeros((self.tr_shape))
    def padimg (self):
        H,W=self.img.shape[0:2]
        tmp_x=self.str*H-H+self.kernal_shape[0]-self.str
        tmp_y=self.str*W-W+self.kernal_shape[1]-self.str
        pad_x,pad_y=[tmp_x//2]*2,[tmp_y//2]*2
        if tmp_x % 2 != 0:
            pad_x[-1]+=1
        if tmp_y % 2 != 0:
            pad_y[-1]+=1
        # print(self.img.shape)
        self.img=np.pad(self.img,(pad_x,pad_y))
        # print(self.img.shape)
        # cv.imshow('nw',self.img)
    def img2mat(self):
        img2=np.zeros((self.kernal_shape[0]*self.kernal_shape[1],self.tr_shape[0]*self.tr_shape[1]))
        # print('img2',img2.shape)
        count=0
        for x in range(0,self.tr_shape[0]):
            for y in range(0,self.tr_shape[1]):
                cur_img=self.img[x*self.str:x*self.str+self.kernal_shape[0],y*self.str:y*self.str+self.kernal_shape[1]].reshape(-1)
                img2[:,count]=cur_img
                count+=1
        return img2
    def max_filter (self):
        if self.padding=='same':
            self.padimg()
        for x in range(0,self.tr_shape[0]):
            for y in range(0,self.tr_shape[1]):
                tmpres=self.img[x*self.str:x*self.str+self.kernal_shape[0],y*self.str:y*self.str+self.kernal_shape[1]].max()
                self.filte_img[x][y]=tmpres
        return self.filte_img
    def min_filter (self):
        if self.padding=='same':
            self.padimg()
        # print(self.tr_shape[0],self.tr_shape[1])
        for x in range(0,self.tr_shape[0]):
            for y in range(0,self.tr_shape[1]):
                tmpres=self.img[x*self.str:x*self.str+self.kernal_shape[0],y*self.str:y*self.str+self.kernal_shape[1]].min()
                self.filte_img[x][y]=tmpres
        return self.filte_img
    def mid_filter (self):
        if self.padding=='same':
            self.padimg()
        # print(self.tr_shape[0],self.tr_shape[1])
        for x in range(0,self.tr_shape[0]):
            for y in range(0,self.tr_shape[1]):
                tmpres=np.median(self.img[x*self.str:x*self.str+self.kernal_shape[0],y*self.str:y*self.str+self.kernal_shape[1]])
                self.filte_img[x][y]=tmpres
        return self.filte_img
    def aver_filter (self):
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=np.ones(self.kernal_shape)/(self.kernal_shape[0]*self.kernal_shape[1])
        # print(filter)
        filter=filter.reshape(1,self.kernal_shape[0]*self.kernal_shape[1])
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        return self.filte_img
    def high_pass(self):
        if self.kernal_shape[0]==3:
            h=np.array([
                [-1,-1,-1],
                [-1,8,-1],
                [-1,-1,-1]
            ])
            
            h=h.astype(np.float64)
            h[1][1]=9*self.high_w-1
            return h/9
        h=np.array([
                [1,1,1],
                [1,8,1],
                [1,1,1]
        ])
        h=h.astype(np.float64)
        h[1][1]=9*self.high_w-1
        s=(self.kernal_shape[0]-1)//2
        for i in range(2,s+1):
            if i==s:
                h=np.pad(h,((1,1),(1,1)),constant_values = (-1,-1))
            else:
                # print('f')
                h=np.pad(h,((1,1),(1,1)),constant_values = (1/i,1/i)) 
        return h/(h.shape[0]*h.shape[1])
    def high_filter(self):
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=self.high_pass()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=(self.filte_img+abs(self.filte_img))/2
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def prewitt(self):
        axis0=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        axis1=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        axis2=np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
        axis3=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        chioc={
            0:axis0,
            1:axis1,
            2:axis2,
            3:axis3
        }
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=chioc.get(self.axis)
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def sobel(self):
        axis0=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        axis1=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        axis2=np.array([[-1,-2,0],[-2,0,2],[0,2,1]])
        axis3=np.array([[0,-2,-1],[2,0,-2],[1,2,0]])
        chioc={
            0:axis0,
            1:axis1,
            2:axis2,
            3:axis3
        }
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=chioc.get(self.axis)
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def laplace(self):
        filter=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def garxy(self):
        filter=np.array([[0,0,0],[0,1,-1],[0,-1,1]])
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def garx2(self):
        filter=np.array([[0,1,0],[0,-2,0],[0,1,0]])
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def gary2(self):
        filter=np.array([[0,0,0],[1,-2,1],[0,0,0]])
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def garx1(self):
        filter=np.array([[-1,0,1]]).T
        # print('filter',filter.shape)
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def gary1(self):
        filter=np.array([[-1,0,1]])
        # print('filter',filter.shape)
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=abs(self.filte_img)
        # print(self.filte_img)
        # print(self.filte_img.dtype)
        return self.filte_img
    def gausskernal(self,sigma=0.8):
        kernal=np.zeros(self.kernal_shape)
        mid=self.kernal_shape[0]//2
        # print(mid)
        sum=0
        for i in range (0,self.kernal_shape[0]):
            for j in range(0,self.kernal_shape[1]):
                kernal[i][j]=math.exp(-((i-mid)**2+(j-mid)**2)/(2*sigma**2))
                sum+=kernal[i][j]
        return kernal/sum
        
    def gauss_filter(self,sigma=0.8):
        filter=self.gausskernal(sigma)
        # print(filter)
        if self.padding=='same':
            self.padimg()
        img2=self.img2mat()
        filter=filter.reshape(1,-1)
        # filter=filter*-1
        self.filte_img=np.matmul(filter,img2).reshape(self.tr_shape)
        self.filte_img=(self.filte_img+abs(self.filte_img))/2
        return self.filte_img
class Filter_w():
    def __init__(self,img,H) -> None:
        self.H=H
        self.img=img
        self.shape=img.shape
        self.center=self.shape[0]//2,self.shape[1]//2
    def my_fft(self):
        img_fft=np.fft.fft2(self.img)
        fshift=np.fft.fftshift(img_fft)
        return fshift
    def my_ifft(self,fft_img):
        ishifft_img=np.fft.ifftshift(fft_img)
        ifft_img=np.fft.ifft2(ishifft_img)
        ifft_img=np.real(ifft_img) 
        return ifft_img
    def ideal_low_filter(self):
        self.mask=np.zeros(self.shape)
        fshift=self.my_fft()
        d=self.H**2
        flag=np.array([((i-self.center[0])**2+(j-self.center[1])**2)<d for i in range(0,self.shape[0]) for j in range(0,self.shape[1])]).reshape(self.shape)
        self.mask[flag]=1
        img_filter_w=fshift*self.mask
        img_filter=self.my_ifft(img_filter_w)
        return img_filter
    def ideal_high_filter(self):
        self.mask=np.zeros(self.shape)
        fshift=self.my_fft()
        d=self.H**2
        flag=np.array([((i-self.center[0])**2+(j-self.center[1])**2)>d for i in range(0,self.shape[0]) for j in range(0,self.shape[1])]).reshape(self.shape)
        self.mask[flag]=1
        img_filter_w=fshift*self.mask
        img_filter=self.my_ifft(img_filter_w)
        return img_filter
    def Butterworth_low_filter(self,n=1):
        fshift=self.my_fft()
        d=self.H**2
        self.mask=np.array([1/(1+(((i-self.center[0])**2+(j-self.center[1])**2)/d)**n) for i in range(0,self.shape[0]) for j in range(0,self.shape[1])]).reshape(self.shape)  
        img_filter_w=fshift*self.mask
        img_filter=self.my_ifft(img_filter_w)
        return img_filter
    def Butterworth_high_filter(self,n=1):
        fshift=self.my_fft()
        d=self.H**2
        self.mask=np.array([11/(1+(d/(((i-self.center[0])**2+(j-self.center[1])**2)+0.0000001))**n)for i in range(0,self.shape[0]) for j in range(0,self.shape[1])]).reshape(self.shape)  
        img_filter_w=fshift*self.mask
        img_filter=self.my_ifft(img_filter_w)
        return img_filter
        
