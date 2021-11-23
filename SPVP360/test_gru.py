# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:12:47 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:18:01 2020

@author: Administrator
"""
from torch import nn
import numpy as np
import torch 
import torch as th
from data_test_gru import VRVideo
import torchvision.transforms as tf
from torch.utils import data as tdata
import torch.optim as optim
from torch.autograd import Variable
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
import visdom
import time
from PIL import Image
from  sp_net  import Final11 ,flownet, Cnn 
from net_gru1  import fore_map ,ConvGRU
from sconv.module import SphereMSE  #SphericalConv 
import os
#import  pdb

def minmaxscaler1(img_map):
    
    img_map=np.maximum(img_map, 0)
    min=th.zeros(img_map.size()[0],1, img_map.size()[2])
    max=th.zeros(img_map.size()[0],1,img_map.size()[2])
    for i11 in range(img_map.size()[0]):
     for i111 in range(img_map.size()[2]):
      min[i11,0,i111]=th.min(img_map[i11,0,i111,:])
      max[i11,0,i111]=th.max(img_map[i11,0,i111,:])
      img_map[i11,0,i111,:]= (img_map[i11,0,i111,:]-min[i11,0,i111])/(max[i11,0,i111]-min[i11,0,i111]+0.00001)
    
    
    return  img_map

    
def minmaxscaler(img_map):
    img_map=np.maximum(img_map, 0)
    min=th.zeros(img_map.size()[0],1,1)
    max=th.zeros(img_map.size()[0],1,1)
    for i11 in range(img_map.size()[0]):
      min[i11]=th.min(img_map[i11])
      max[i11]=th.max(img_map[i11])
      img_map[i11]= (img_map[i11]-min[i11])/(max[i11]-min[i11]+0.00001)
      
     

    #img_map=np.maximum(img_map, 0)
    #img_map=abs(img_map)
    #min=np.amin(img_map)
    #max=np.amax(img_map)
    #img_map= (img_map-min)/(max-min+0.00001)
    #np.where(img_map > 0.5,  img_map, 0)
    #img_map=np.maximum(img_map, 0.5 )
    return  img_map 

def minmaxscaler2(img_map):
    img_map=np.maximum(img_map, 0)
    #img_map=abs(img_map)
    min=np.amin(img_map)
    max=np.amax(img_map)
    img_map= (img_map-min)/(max-min+0.00001)
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
    data_x=[]
    data_y=[]
    for i in range(len(data)- seg_len): #-look_back
         data_x.append(data[i:i+ seg_len])
         data_y.append(target_data[i+seg_len])     
    return data_x, data_y

def creatdataset1(data):
    #look_back=5
    seg_len=2
    data_x=[]
    #data_y=[]
    for i in range(len(data)- seg_len): #-look_back
         data_x.append(data[i:i+ seg_len])
         #data_y.append(target_data[i+seg_len])     
    return data_x
  
def creatdataset2(target_data):
    #look_back=5
    seg_len=2
    #data_x=[]
    data_y=[]
    for i in range(len(target_data)- seg_len): #-look_back
         data_y.append(target_data[i+seg_len])     
    return  data_y  

def fusion(out_st,out_gru,h,w):
   
    max_st=th.zeros(out_st.size()[0],1,1)
    max_gru=th.zeros(out_st.size()[0],1,1)
    mean_st=th.zeros(out_st.size()[0],1,1)
    mean_gru=th.zeros(out_st.size()[0],1,1)
    cou_w=int(out_st.size()[3]/w)
    cou_h=int(out_st.size()[2]/h)
    max_rst=th.zeros(out_st.size()[0],cou_h,cou_w)
    max_rgru=th.zeros(out_st.size()[0],cou_h,cou_w)
    out=th.zeros(out_st.size()[0],1,out_st.size()[2],out_st.size()[3])
                                     
    for i11 in range(out_st.size()[0]):
      max_st[i11]=th.max(out_st[i11])
      max_gru[i11]=th.max(out_gru[i11])
      for j in range(cou_h):
        for k in range(cou_w):
          max_rst[i11,j,k]= th.max(out_st[i11,0,j*h:(j+1)*h-1,k*w:(k+1)*w-1] ) 
          max_rgru[i11,j,k]=th.max( out_gru[i11,0,j*h:(j+1)*h-1,k*w:(k+1)*w-1])  
       
      mean_st[i11]=th.mean(max_rst)
      mean_gru[i11]=th.mean(max_rgru)
      out[i11]=out_st[i11]*(max_st[i11]-mean_st[i11])*(max_st[i11]-mean_st[i11])+out_gru[i11]*(max_gru[i11]-mean_gru[i11])*(max_gru[i11]-mean_gru[i11])
      
    return out          
            

def count_params(model, bs, channel,input_size_h,input_size_w):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))
 
    # 计算模型的计算量
    calc_flops(model,  bs, channel,input_size_h,input_size_w)
 
    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
 
    print('The network has {} params.'.format(params))
 
 
# 计算模型的计算量
def calc_flops(model, bs, channel,input_size_h,input_size_w):
 
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv.append(flops)
 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
 
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
 
    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    USE_GPU=True
    if '0.4.' in torch.__version__:
        if  USE_GPU:
            input = torch.cuda.FloatTensor(torch.rand(bs, channel, input_size_h, input_size_w).cuda())
        else:
            input = torch.FloatTensor(torch.rand(bs, channel, input_size_h, input_size_w))
    else:
        input = Variable(torch.rand(bs, channel, input_size_h, input_size_w), requires_grad=True)
    _ = model(input)
 
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
 
    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / bs))

#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
log_dir1='st-spcnncbam62.pth.tar'
log_dir='net_gru.pth.tar'


#E:/experimental result1/1/1/test/
def test(
        root='E:/experimental result1/1/1/test/',  #E:/360_VRvideo','F:/360VRvideo', #360_Saliency_dataset_2018ECCV
        bs=4, #28
        lr=0.005,
        epochs1=63,
        epochs=119,
        clear_cache=False,
        plot_server='http://127.0.0.1',
        plot_port=8097,
        save_interval=1,
        resume=True,
        resume_saliency=True,  #True False
        height =112,
        width = 224,
        start_epoch=0,
        exp_name='final',
        step_size=10,
        test_mode=True #False,
        
):
    tic=time.time()
    print(tic)
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((height, width)), #128, 256  #shuffle=true 数据集打乱     
        tf.ToTensor()])
    
    tar_transform = tf.Compose([
        tf.Resize((int(height/2), int(width/2))), #128, 256  #shuffle=true 数据集打乱     
        tf.ToTensor()
    ])
    dataset = VRVideo(root, height, width, 15, frame_interval=5, 
                      cache_gt= True, transform=transform,
                      tar_transform=tar_transform,gaussian_sigma=np.pi/20,
                      kernel_rad=np.pi/7)
    #print(dataset)  #128, 256,
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle= False , num_workers=0, pin_memory=False )#True TrueTrue
    #print(type(loader))
    
    
    
    model_cnn=Cnn()
    model_flo=flownet()
    model_final = Final11()  
    
    model_cnn= nn.DataParallel(model_cnn).cuda()
    model_flo=nn.DataParallel(model_flo).cuda()
    model_final =nn.DataParallel(model_final).cuda()
    
    if resume:
        checkpoint = th.load(log_dir1,map_location=lambda storage, loc: storage)
        model_cnn.load_state_dict(checkpoint['model_cnn_state_dict'])
        model_flo.load_state_dict(checkpoint['model_flo_state_dict'])
        model_final.load_state_dict(checkpoint['model_state_dict'])       
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
        
    gpu_model_cnn = model_cnn#.module
    gpu_model_flo = model_flo#.module
    gpu_model_final = model_final#.module
    
    
    use_gpu = th.cuda.is_available()
    if use_gpu:
            dtype = th.cuda.FloatTensor # computation in GPU
    else:
            dtype = th.FloatTensor
    kernel_size = (3,3)  
    modelGRU = ConvGRU(input_size=(56, 112),
                    input_dim=1,
                    hidden_dim=[64, 64],
                    kernel_size=(3,3),
                    num_layers=2,
                    dtype=dtype,
                    batch_first=True,
                    bias = False,
                    return_all_layers = False)
    
    
    modelGRU=modelGRU.cuda()
    model=fore_map() 
    model=model.cuda()
    
    if os.path.exists(log_dir):
        checkpoint = th.load(log_dir)
        modelGRU.load_state_dict(checkpoint['modelGRU_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
     
     
    #params_cnn=count_params(gpu_model_cnn,1,3,224,448)
    #params_flo=count_params(gpu_model_flo,1,6,224,448)
    #params_final=count_params(gpu_model_final,1,192,56,112)
    #params_flo=count_params(modelGRU,1,3,56,112)
    #params_final=count_params(model,1,64,56,112)

    
    data_loss=[]
    seg_len=2
    
    for epoch in trange(0,1):#start_epoch, epochs, desc='epoch'
    #    train_loss=0.
        tic = time.time()
        for i, (img1_batch,img2_batch,input_m_batch,input_m2_batch,input_m3_batch,input_m4_batch,input_m5_batch,input_m6_batch,input_m7_batch,input_m8_batch,input_m9_batch,input_m10_batch,input_m11_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
      #input_m12_batch,input_m13_batch
      #input_m9_batch,input_m10_batch 
            data_time = time.time() - tic
            tic = time.time()
            
        #ST-SPCNN    
            img1_var = Variable(img1_batch,requires_grad=False).cuda()
            img2_var = Variable(img2_batch,requires_grad=False).cuda()
          
            out_cnn    = gpu_model_cnn(img1_var)
            
            input_flo = th.cat([img1_var,img2_var], dim=1)
            out_flo = gpu_model_flo(input_flo)
           
            input_final= th.cat([out_cnn, out_flo], dim=1)
            out_final  = gpu_model_final( input_final) 
            
            st_cnn_time = time.time() - tic
            tic = time.time()
        #SP-GRU    
            
            img_x_batch1=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) #5 表示bitch size， 3表示seq_len
            img_x_batch2=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch3=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch4=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch5=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch6=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch7=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2))  
            img_x_batch8=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch9=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch10=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            img_x_batch11=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            #img_x_batch12=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2)) 
            #img_x_batch13=th.zeros(int(input_m_batch.size()[0]-seg_len),seg_len,1,int(height/2),int(width/2))  
            
            #img_y_batch1=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            out_stcnn=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency2=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency3=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency4=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency5=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency6=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency7=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency8=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency9=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency10=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency11=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency12=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
            # out_saliency13=th.zeros(int(input_m_batch.size()[0]-seg_len),1,int(height/2),int(width/2))
           
            
            img_x1 =creatdataset1(input_m_batch) #
            img_x2 =creatdataset1(input_m2_batch) 
            img_x3 =creatdataset1(input_m3_batch) 
            img_x4 =creatdataset1(input_m4_batch) 
            img_x5 =creatdataset1(input_m5_batch) 
            img_x6 =creatdataset1(input_m6_batch) 
            img_x7 =creatdataset1(input_m7_batch) 
            img_x8 =creatdataset1(input_m8_batch) 
            img_x9 =creatdataset1(input_m9_batch) 
            img_x10 =creatdataset1(input_m10_batch) 
            img_x11=creatdataset1(input_m11_batch) 
            #img_x12 =creatdataset1(input_m12_batch) 
            #img_x13 =creatdataset1(input_m13_batch)
            
            out_st=creatdataset2(out_final.data.cpu()) 
            
            for i1 in range(int(input_m_batch.size()[0]-seg_len)):
               out_stcnn[i1]=out_st[i1]
               img_x_batch1[i1]= img_x1[i1] 
               img_x_batch2[i1]= img_x2[i1] 
               img_x_batch3[i1]= img_x3[i1] 
               img_x_batch4[i1]= img_x4[i1] 
               img_x_batch5[i1]= img_x5[i1] 
               img_x_batch6[i1]= img_x6[i1] 
               img_x_batch7[i1]= img_x7[i1] 
               img_x_batch8[i1]= img_x8[i1] 
               img_x_batch9[i1]= img_x9[i1] 
               img_x_batch10[i1]= img_x10[i1] 
               img_x_batch11[i1]= img_x11[i1] 
               #img_x_batch12[i1]= img_x12[i1] 
               #img_x_batch13[i1]= img_x13[i1] 
               
               #img_y_batch1[i1]= img_y[i1]
               
            #img_x_batch1=th.squeeze(img_x_batch1)
            
            img_x_int1=Variable(img_x_batch1,requires_grad=False).cuda()
            img_x_int2=Variable(img_x_batch2,requires_grad=False).cuda()
            img_x_int3=Variable(img_x_batch3,requires_grad=False).cuda()
            img_x_int4=Variable(img_x_batch4,requires_grad=False).cuda()
            img_x_int5=Variable(img_x_batch5,requires_grad=False).cuda()
            img_x_int6=Variable(img_x_batch6,requires_grad=False).cuda()
            img_x_int7=Variable(img_x_batch7,requires_grad=False).cuda()
            img_x_int8=Variable(img_x_batch8,requires_grad=False).cuda()
            img_x_int9=Variable(img_x_batch9,requires_grad=False).cuda()
            img_x_int10=Variable(img_x_batch10,requires_grad=False).cuda()
            img_x_int11=Variable(img_x_batch11,requires_grad=False).cuda()
            #img_x_int12=Variable(img_x_batch12,requires_grad=False).cuda()
            
            #img_x_int13=Variable(img_x_batch13,requires_grad=False).cuda()
            
            
            #img_y_int=Variable(img_y_batch1).cuda()
            
            layer_output_list, last_state_list1 = modelGRU(img_x_int1)
            layer_output_list, last_state_list2 = modelGRU(img_x_int2)
            layer_output_list, last_state_list3 = modelGRU(img_x_int3)
            layer_output_list, last_state_list4 = modelGRU(img_x_int4)
            layer_output_list, last_state_list5 = modelGRU(img_x_int5)
            layer_output_list, last_state_list6 = modelGRU(img_x_int6)
            layer_output_list, last_state_list7 = modelGRU(img_x_int7)
            layer_output_list, last_state_list8 = modelGRU(img_x_int8)
            
            layer_output_list, last_state_list9 = modelGRU(img_x_int9)
            layer_output_list, last_state_list10 = modelGRU(img_x_int10)
            layer_output_list, last_state_list11 = modelGRU(img_x_int11)
            #layer_output_list, last_state_list12 = modelGRU(img_x_int12)
            #layer_output_list, last_state_list13 = modelGRU(img_x_int13)
            
            out1=model(last_state_list1[0][0])
            out2=model(last_state_list2[0][0])
            out3=model(last_state_list3[0][0])
            out4=model(last_state_list4[0][0]) 
            out5=model(last_state_list5[0][0]) 
            out6=model(last_state_list6[0][0]) 
            out7=model(last_state_list7[0][0]) 
            out8=model(last_state_list8[0][0]) 
            out9=model(last_state_list9[0][0])  
            out10=model(last_state_list10[0][0])  
            out11=model(last_state_list11[0][0])  
            #out12=model(last_state_list12[0][0])  
            #out13=model(last_state_list13[0][0]) 
            
           
            #viz.images( minmaxscaler2(out1.data.cpu().numpy()), win='out_11') 
            
            out_gru=out1.data.cpu()+out2.data.cpu()+out3.data.cpu()+out4.data.cpu()+out5.data.cpu()+out6.data.cpu()+out7.data.cpu()+out8.data.cpu()+out9.data.cpu()+out10.data.cpu()+out11.data.cpu()#+out12.data.cpu()+out13.data.cpu()
            
            out_gru1=th.from_numpy(minmaxscaler2(out_gru.numpy())*0.6)
            
            
            viz.images(out_gru1, win='out_gru11') 
            
            out_stf=minmaxscaler(out_stcnn)
            
           # viz.images(out_stf.numpy(), win='out_st11') 
            
            out=fusion(out_stf,out_gru1,7,14)
            
            gru_time = time.time() - tic
            tic = time.time()
            viz.images(out.numpy(), win='out_11') 
            
            #fwd_time = time.time() - tic
            #tic = time.time()
                
            for i1  in   range(out1.size()[0]):
               #output=minmaxscaler(out)
               out_wr=np.squeeze((out.numpy()*255).astype(np.uint8),1)
               target_img=Image.fromarray(out_wr[i1])
               target_img.save('E:/experimental result/1/'+str(i*bs+i1+1)+'.png')
        
            
            write_time = time.time() - tic
            tic = time.time()
            #print(time.time())
            #print(gru_time)
            
            log_file1 = open('E:/experimental result/1/data_use11.txt','a+')
            msg ='{:d},time: data={},st-cnn={},gru={},write={},total={}'.format(i,data_time,st_cnn_time,gru_time,write_time,st_cnn_time+gru_time)
            print(msg,file=log_file1,flush=True)
            
            del img1_batch,img2_batch,img1_var,img2_var,input_flo,input_final, out_cnn,out_flo,out_final,
            out_st, out_stcnn,out_stf,
            input_m_batch,input_m2_batch,input_m3_batch,input_m4_batch,input_m5_batch#,input_m6_batch,input_m7_batch,
            img_x1,img_x2,img_x3,img_x4,img_x5#,img_x6,img_x7,img_x8,
            img_x_int1, img_x_int2, img_x_int3, img_x_int4, img_x_int5#,img_x_int6, img_x_int7, img_x_int8,
            layer_output_list, last_state_list1,last_state_list2,last_state_list3,last_state_list4,last_state_list5,
            #last_state_list6,last_state_list7,last_state_list8,
            out1, out2,out3,out4,out5#,out6,out7,out8,
            out,out_gru,out_gru1
            
            th.cuda.empty_cache()
        
            
        
if __name__ == '__main__':
    
    Fire(test)
  
