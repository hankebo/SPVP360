# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:49:22 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:01:55 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:10:54 2020

@author: Administrator
"""

import numpy as np
import torch as th
#import torch.utils.data as data
from PIL import Image
import os
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms, utils
from scipy import signal
from sconv.functional.sconv import spherical_conv
from tqdm import tqdm
import numbers
import cv2
from functools import lru_cache
from random import Random
import pdb
import visdom


#root= 'C:/Users/Admin/Desktop/HL/1'  #'E:/360_VRvideo'
frame_h=112
frame_w=224  #448
#video_train=4
BATCH_SIZE=1

data_transform = transforms.Compose([
        transforms.Resize((112, 224)), #  224, 448
        transforms.ToTensor()
    ])

target_transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize((62,118)),  #     126,238  124,236
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

def minmaxscaler(img_map):
    img_map=np.maximum(img_map, 0)
    #img_map=abs(img_map)
    min=np.amin(img_map)
    max=np.amax(img_map)
    img_map= (img_map-min)/(max-min+0.00001)
    #np.where(img_map > 0.5,  img_map, 0)
    #img_map=np.maximum(img_map, 0.5 )
    return  img_map 
#res =pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))
#result = np.mat(res)
#scio.savemat('E:/360_Saliency_dataset_2018ECCV/res.mat', {'result': res})

def default_loader(root):
    return Image.open(root).convert('RGB')

class VRVideo(Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, tar_transform=None,
                 train=True, loader=default_loader,gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60),
                 cache_gt=True, rnd_seed=367643):#
        self.root=root
        self.frame_interval = frame_interval
        self.transform = transform
        self.target_transform = tar_transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size   #指的是图像一部分区间
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.loader=loader
        rnd = Random(rnd_seed)   #得到不变的随机数列  是一个种子，rnd_seed对应一个随机数里面的一个数，固定不变的

        # load target
        self.vinfo = pickle.load(open(os.path.join(self.root, 'vinfo.pkl'), 'rb'))

        vid ='245'
        
        self.data = []
        self.target = []
        self.input_m=[]
        self.i2v = {}
        self.v2i = {}
        self.i2v1 = {}
        self.v2i1 = {}
        #for vid in vset:
        obj_path = os.path.join(root, vid)  #路径融合
            # fcnt = 0
        frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
        frame_list.sort()
          
        for frame in frame_list:
                fid = frame[:-4]  #0001  图片的名字字符串
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.input_m.append(self.vinfo[vid][fid][0:1]) 
                self.target.append(self.vinfo[vid][fid]) 
        
        for i in range(len(self.data)):
              fr_data1 =  self.data[0]  # 获取第一个座位的学生 student1
              fr_data =   self.data[1:]  # 让 student1 暂时离开，后面的学生座位都进一位。
              fr_data.append(fr_data1)
        self.fr_data=  fr_data

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img)
            
        img1 = Image.open(open(self.fr_data[item], 'rb'))
        #img = img.resize((self.frame_w, self.frame_h))
        if self.transform is not None:
            img1 = self.transform(img1)
        else:
            img1 = np.array(img1)
       
    
       
        input_m= self._get_sal_map(item)
        input_m2=self._get_sal_map2(item)
        input_m3= self._get_sal_map3(item)
        input_m4= self._get_sal_map4(item)
        input_m5= self._get_sal_map5(item)
        input_m6= self._get_sal_map6(item)
        input_m7= self._get_sal_map7(item)
        input_m8= self._get_sal_map8(item)
        input_m9= self._get_sal_map9(item)
        input_m10= self._get_sal_map10(item)
        input_m11= self._get_sal_map11(item)
        #input_m12= self._get_sal_map12(item)
        #input_m13= self._get_sal_map13(item)
        #input_m14= self._get_sal_map14(item)
        
        
       # target = self._get_salency_map(item)

        #if self.train:img, img1,
        return img, img1,input_m,input_m2, input_m3,input_m4, input_m5,input_m6, input_m7,input_m8,input_m9,input_m10,input_m11#,input_m12,input_m13,input_m14               
    

    def __len__(self):
        return len(self.data)
    
    def _get_sal_map(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-1.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
        input_m = np.zeros((int(self.frame_h/2), int(self.frame_w/2)))
        for x_norm1, y_norm1 in self.input_m[item]:
            x, y = min(int(x_norm1 * int(self.frame_w/2) + 0.5), int(self.frame_w/2) - 1), min(int(y_norm1 * int(self.frame_h/2) + 0.5), int(self.frame_h/2) - 1)
            input_m[y, x] = 100
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            input_map = spherical_conv(
                th.from_numpy(
                    input_m.reshape(1, 1, *input_m.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, int(self.frame_h/2), int(self.frame_w/2))
        else:
            input_map = spherical_conv(
                th.from_numpy(
                    input_m.reshape(1, 1, *input_m.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, int(self.frame_h/2), int(self.frame_w/2))
        if item >= 0 and self.cache_gt:
            np.save(cfile1, input_map.data.cpu().numpy() ) #len(self.target[item]))

        return input_map.data.float()  #len(self.target[item])
    
    
    def _get_sal_map2(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-2.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
                
                
    def _get_sal_map3(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-3.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()         
    
    def _get_sal_map4(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-4.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
     
    def _get_sal_map5(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-5.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float() 

    def _get_sal_map6(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-6.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()

    def _get_sal_map7(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-7.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
    
    def _get_sal_map8(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-8.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
                
    def _get_sal_map9(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-9.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()


    def _get_sal_map10(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-10.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float() 
     
     
     
    def _get_sal_map11(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-11.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float() 
     
     
    def _get_sal_map12(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-12.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()  
     

    def _get_sal_map13(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-13.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float() 

    
    def _get_sal_map14(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '-14.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float() 




   
    
    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1,  int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((int(self.frame_h/2), int(self.frame_w/2)))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * int(self.frame_w/2) + 0.5), int(self.frame_w/2) - 1), min(int(y_norm * int(self.frame_h/2) + 0.5), int(self.frame_h/2) - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1,  int(self.frame_h/2), int(self.frame_w/2))
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1,  int(self.frame_h/2), int(self.frame_w/2))
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache1(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile1 = self.data[item][:-4] + '.npy'
            if os.path.isfile(cfile1):
                print('remove {}'.format(cfile1))
                os.remove(cfile1)
        return self
    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
            self._get_sal_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


#video_train=1
#root='C:/Users/Admin/Desktop/HL/test'

#dataset = VRVideo(root,112, 224, video_train, frame_interval=5,transform=data_transform, cache_gt=True,
 #                gaussian_sigma=np.pi/20, kernel_rad=np.pi/7, loader=default_loader)
#data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
#plot_server='http://127.0.0.1'
#plot_port='8097'
#exp_name='final'
#viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

#if __name__ == '__main__':
 
 #root='E:/360_VRvideo', input_m3,input_m4
 # for i, ( img, img1,input_m,input_m2,input_m3) in tqdm(enumerate(data_loader), desc='batch', total=len(data_loader)):#last_batch ,img2_batch , target_batch
          
  #  viz.images( minmaxscaler(input_m.cpu().numpy()), win='gt-in1')    
   # viz.images( minmaxscaler(input_m2.cpu().numpy()), win='gt-in2')   


class VRVideosaliency(Dataset):
    def __init__(self, root,root1, frame_h, frame_w, video_train, 
                 frame_interval=1, transform=None, tar_transform=None,
                 train=True, loader=default_loader,gaussian_sigma=np.pi/20,
                 kernel_rad=np.pi/7, kernel_size=(30, 60),
                 cache_gt=True, rnd_seed=367643):#
        self.root=root
        self.root=root1
        self.frame_interval = frame_interval
        self.transform = transform
        self.target_transform = tar_transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size   #指的是图像一部分区间
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.loader=loader
        rnd = Random(rnd_seed)   #得到不变的随机数列  是一个种子，rnd_seed对应一个随机数里面的一个数，固定不变的

        # load target
        self.vinfo = pickle.load(open(os.path.join(self.root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(self.root), desc='scanning dir'):
            if os.path.isdir(os.path.join(self.root, vid)):
                vset.append(vid)
        vset.sort()
            
            
        #assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))
        
        vset = vset_train if train else vset_val
        
        self.data = []
        self.data1 = []
        self.target = []
        self.input_m=[]
        self.i2v = {}
        self.v2i = {}
        self.i2v1 = {}
        self.v2i1 = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)  #路径融合
            obj_path1 = os.path.join(root1, vid) 
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            
            frame_list1 = [frame1 for frame1 in os.listdir(obj_path1) if frame1.endswith('.jpg')]
            frame_list1.sort()
          
            for frame in frame_list:
                fid = frame[:-4]  #0001  图片的名字字符串
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.input_m.append(self.vinfo[vid][fid][0:5]) 
                self.target.append(self.vinfo[vid][fid]) 
               
            for frame1 in frame_list1:
                fid1 = frame1[:-4]  #0001  图片的名字字符串
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data1)] = (vid, fid1)
                self.v2i[(vid, fid1)] = len(self.data1)
                self.data1.append(os.path.join(obj_path, frame1))
                #self.input_m.append(self.vinfo[vid][fid][0:5]) 
                #self.target.append(self.vinfo[vid][fid]) 
                
            #frame_list1 = [frame for frame in os.listdir(obj_path) if frame.endswith('.png')]
            #frame_list1.sort()
            #for frame1 in frame_list1:
                #fid1 = frame1[:-4]#0001  图片的名字字符
                #self.i2v1[len(self.target)] = (vid, fid1)
                #self.v2i1[(vid, fid1)] = len(self.target)
                #self.target.append(os.path.join(obj_path, frame1))        # fcnt = 0

        #self.target.append([(0.5, 0.5)])
        
        for i in range(len(self.data)):
              fr_data1 =  self.data[0]  # 获取第一个座位的学生 student1
              fr_data =   self.data[1:]  # 让 student1 暂时离开，后面的学生座位都进一位。
              fr_data.append(fr_data1)
        self.fr_data=  fr_data

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img)
            
        img1 = Image.open(open(self.fr_data[item], 'rb'))
        #img = img.resize((self.frame_w, self.frame_h))
        if self.transform is not None:
            img1 = self.transform(img1)
        else:
            img1 = np.array(img1)
       
        saliency = Image.open(open(self.data1[item], 'rb'))
     
        if  self.target_transform  is not None:
             saliency  =  self.target_transform ( saliency )
        else:
            saliency  = np.array( saliency)
       
       # vid, fid = self.i2v[item]
       
        #tar1 = Image.open(open(self.target[item], 'rb'))
        #if self.transform is not None:
        #    tar1 = self.transform(tar1)
        #el:
        ##    tar1 = np.array(tar1)
        input_m= self._get_sal_map(item)
        target = self._get_salency_map(item)

        if self.train:
            return   img, img1,input_m,  target ,saliency   #last ,img1, 
        else:
            return  img, img1, input_m, target,self.data[item], saliency #self.data[item],  tar1  #get # last img1,

    def __len__(self):
        return len(self.data)
    
    def _get_sal_map(self, item, use_cuda=False):
        cfile1 = self.data[item][:-4] + '.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile1):
                input_map = th.from_numpy(np.load(cfile1)).float()
                assert input_map.size() == (1, int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile1)).float()
        input_m = np.zeros((int(self.frame_h/2), int(self.frame_w/2)))
        for x_norm1, y_norm1 in self.input_m[item]:
            x, y = min(int(x_norm1 * int(self.frame_w/2) + 0.5), int(self.frame_w/2) - 1), min(int(y_norm1 * int(self.frame_h/2) + 0.5), int(self.frame_h/2) - 1)
            input_m[y, x] = 100
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            input_map = spherical_conv(
                th.from_numpy(
                    input_m.reshape(1, 1, *input_m.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, int(self.frame_h/2), int(self.frame_w/2))
        else:
            input_map = spherical_conv(
                th.from_numpy(
                    input_m.reshape(1, 1, *input_m.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, int(self.frame_h/2), int(self.frame_w/2))
        if item >= 0 and self.cache_gt:
            np.save(cfile1, input_map.data.cpu().numpy() ) #len(self.target[item]))

        return input_map.data.float()  #len(self.target[item])
    
    
    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1,  int(self.frame_h/2), int(self.frame_w/2))
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((int(self.frame_h/2), int(self.frame_w/2)))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * int(self.frame_w/2) + 0.5), int(self.frame_w/2) - 1), min(int(y_norm * int(self.frame_h/2) + 0.5), int(self.frame_h/2) - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1,  int(self.frame_h/2), int(self.frame_w/2))
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1,  int(self.frame_h/2), int(self.frame_w/2))
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache1(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile1 = self.data[item][:-4] + '.npy'
            if os.path.isfile(cfile1):
                print('remove {}'.format(cfile1))
                os.remove(cfile1)
        return self
    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
            self._get_sal_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self   
#dataset = VRVideo(root,112, 224, video_train, frame_interval=5,transform=data_transform, cache_gt=True,
 #                gaussian_sigma=np.pi/20, kernel_rad=np.pi/7, loader=default_loader)
#data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
