# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:18:01 2020

@author: Administrator
"""
from torch import nn
import numpy as np
import torch as th
from data_gru import VRVideo
import torchvision.transforms as tf
from torch.utils import data as tdata
import torch.optim as optim
from torch.autograd import Variable
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
import visdom
import time

#from  sp_net  import Final ,flownet, Cnn ,Cnn_map,flo_map 
from net_spgru  import fore_map
#from  sp_net1  import Final  # spherical_unet
from sconv.module import SphereMSE  #SphericalConv 
import os

def minmaxscaler1(img_map):
    img_map=np.maximum(img_map, 0)
    min=th.zeros(img_map.size()[0],1,1)
    max=th.zeros(img_map.size()[0],1,1)
    for i11 in range(img_map.size()[0]):
      min[i11]=th.min(img_map[i11])
      max[i11]=th.max(img_map[i11])
      img_map[i11]= (img_map[i11]-min[i11])/(max[i11]-min[i11]+0.00001)
      
    #img_map=abs(img_map)
    #min=np.amin(img_map)
    #max=np.amax(img_map)
    #img_map= (img_map-min)/(max-min+0.00001)
    #np.where(img_map > 0.5,  img_map, 0)
    #img_map=np.maximum(img_map, 0.5 )
    return  img_map 
  
def minmaxscaler(img_map):
    img_map=np.maximum(img_map, 0)
    #img_map=abs(img_map)
    min=np.amin(img_map)
    max=np.amax(img_map)
    img_map= (img_map-min)/(max-min+0.00000001)
    #np.where(img_map > 0.5,  img_map, 0)
    #img_map=np.maximum(img_map, 0.5 )
    return  img_map 
    
def minmaxsmap(img_m):
    img_m=np.abs(img_m)
    #img_map=abs(img_map)
    min=np.amin(img_m)
    max=np.amax(img_m)
    img_m= (img_m-min)/(max-min+0.00001)
    #np.where(img_map > 0.5,  img_map, 0)
    #img_map=np.maximum(img_map, 0.5 )
    return  img_m 


def creatdataset(data,target_data):
    #look_back=5
    seg_len=3
    #in_user=5
    #x1=th.zeros(int(len(data)/look_back),in_user,1,112,224 ) # data.size()[1:]
    #x2=th.zeros(int(len(data)/look_back-seg_len),in_user,seg_len,1,112,224)
    data_x=[]
    data_y=[]
    #data_saliency=[]
    #x= th.split(data, in_user,dim=0)
    #for j in range(int(len(data)/look_back)):
       # x1[j]=x[j]
    #x1 = x1.permute(1, 0, 2, 3, 4)
    for i in range(len(data)- seg_len): #-look_back
     #for j in range(5):
         data_x.append(data[i:i+ seg_len])
         #x2[i]=x1[:, i:i+ seg_len, :, :, :]
         data_y.append(target_data[i+seg_len])
         #data_saliency.append(saliency[i+seg_len])
    #x2 = x2.permute(1, 0, 2, 3, 4,5) 
#   for k in range(in_user):    
        #data_x.append(x2[k])  #data[i:i+look_back] split 1[:, i:i+ seg_len, :, :, :]
      
    return data_x, data_y#, data_saliency
    
lr1=0.01
def adjust_learning_rate(epoch):
    if epoch <= 20:  # 32k iterations
      lr=lr1
    elif epoch > 20 and epoch <= 50:  # 48k iterations
      lr=lr1/10
    elif epoch > 50 and epoch <= 70:  # 48k iterations
      lr=lr1/100
    elif epoch > 70 and epoch <= 90:
      lr=lr1/1000
    else:
      lr=lr1/10000
    return lr    
    
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def train(
        root='C:/Users/Admin/Desktop/HL/360video_our_all',  #
        bs=20, #28
        lr=0.01,
        epochs=120,
        clear_cache=False,
        plot_server='http://127.0.0.1',
        plot_port=8097,
        save_interval=1,
        resume=False,
        resume_saliency=True,  #True False
        height =112,
        width = 224,
        start_epoch=0,
        exp_name='final',
        step_size=10,
        test_mode=True #False,
        
):
    
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((height, width)), #128, 256  #shuffle=true 数据集打乱     
        tf.ToTensor()])
    
    tar_transform = tf.Compose([
        tf.Resize((int(height/2), int(width/2))), #128, 256  #shuffle=true 数据集打乱     
        tf.ToTensor()
    ])
    dataset = VRVideo(root, height, width, 14, frame_interval=5, 
                      cache_gt= True, transform=transform,
                      tar_transform=tar_transform,gaussian_sigma=np.pi/20,
                      kernel_rad=np.pi/7)
    #print(dataset)  #128, 256,
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle= False , num_workers=0, pin_memory=False )#True TrueTrue
    #print(type(loader)) 
    
   
    
    model=fore_map() 
    model=model.cuda()
    #modelGRU=modelGRU.cuda()
    
    #optimizerGRU = optim.SGD(modelGRU.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)  #优化器
    criterion = SphereMSE(int(height/2),int( width/2)).float().cuda()    #损失函数128,256
    
    log_dir='net_spgru.pth.tar'
    if os.path.exists(log_dir):
        checkpoint = th.load(log_dir)
        #modelcell.load_state_dict(checkpoint['modelcell_state_dict'])
        #modelGRU.load_state_dict(checkpoint['modelGRU_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
   
    log_file = open( 'C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/gru/spgru_loss.txt', 'a+')

    
        
    data_loss=[]
    seg_len=3
    for epoch in trange(start_epoch, epochs, desc='epoch'):
        train_loss=0.
        tic = time.time()
        lr=adjust_learning_rate(epoch)
        #optimizerGRU = optim.SGD(modelGRU.parameters(), lr, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)
        for i, (img1_batch,img2_batch,input_m_batch,target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):#last_batch ,img2_batch , target_batch
        
           
            data_time = time.time() - tic
            tic = time.time()
            
            img_x_batch1=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) #5 表示bitch size， 3表示seq_len
            img_y_batch1=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            #out_saliency=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            #out1=model1(img1_var)
            #out_saliency=out1[seg_len:input_m_batch.size()[0]]
            
            img_x,img_y =creatdataset(input_m_batch,target_batch) #
            
            for i1 in range(int(input_m_batch.size()[0]-seg_len)):
               img_x_batch1[i1]= img_x[i1] 
               img_y_batch1[i1]= img_y[i1]
               #out_saliency[i1]=img_saliency[i1]
            #img_x_batch1=th.squeeze(img_x_batch1)
            
            img_x_int=Variable(img_x_batch1).cuda()
            img_y_int=Variable(img_y_batch1).cuda()
            
            #layer_output_list,last_state_list = modelGRU(img_x_int)
            out=model(img_x_int) #last_state_list1#+m*saliency
           
        
            optimizer.zero_grad()
            #out=model(img1_var)
            #out = pmodel(img1_var,img2_var) #, ,last_var ,img2_var
            
            loss = criterion(out,img_y_int)
            fwd_time = time.time() - tic
            tic = time.time()
            loss.backward()
            
            #optimizerGRU.step()
            optimizer.step()
           
            train_loss += loss.data[0]#*(input_m_batch.size()[0]-seg_len) 
        
            viz.images( minmaxscaler(input_m_batch.cpu().numpy()), win='gt') 
            #viz.images(minmaxscaler(out_saliency.data.cpu().numpy()), win='in')# * 10 abs(out.data.cpu().numpy()) np.maximum
          #  viz.images(minmaxscaler(target_batch.cpu().numpy()), win='gt1') 
            viz.images(minmaxscaler(out.data.cpu().numpy()), win='out_gru') #out.data.cpu().numpy()*1000
            #viz.images(minmaxscaler(OUT.data.cpu().numpy()), win='ot1')
            #viz.text(msg, win='log')
           
            tic = time.time()
            
            del input_m_batch,target_batch,img_x_int,img_y_int,img1_batch,img2_batch,out,img_x,img_y,loss,
            
            th.cuda.empty_cache()
            
        if (epoch + 1) % save_interval == 0:
            th.save({'epoch': epoch,
                     #'modelcell_state_dict': modelcell.state_dict(),
                     #'modelGRU_state_dict': modelGRU.state_dict(),
                     'model_state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()
                    },'net_spgru.pth.tar') # + exp_name  
        data_loss.append(train_loss)
        train_epoch_loss='{:d}   {:.6f}'.format(epoch,train_loss)
        print(train_epoch_loss,file=log_file,flush=True)
            
        
if __name__ == '__main__':
    
    Fire(train)
  
