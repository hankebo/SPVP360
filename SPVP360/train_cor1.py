from torch import nn
import numpy as np
import torch as th
from data3 import VRVideo
import torchvision.transforms as tf
from torch.utils import data as tdata
import torch.optim as optim
from torch.autograd import Variable
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
import visdom
import time
import os
#import copy
from  sp_net  import Final11 ,flownet, Cnn # ,Cnn_map,flo_map # spherical_unet
from sconv.module import SphereMSE #,SFLoss

#import  pdb

def minmaxscaler2(img_map):
    
    img_map=np.maximum(img_map, 0)
    min=th.zeros(img_map.size()[0],1, img_map.size()[3])
    max=th.zeros(img_map.size()[0],1,img_map.size()[3])
    for i11 in range(img_map.size()[0]):
     for i111 in range(img_map.size()[3]):
      min[i11,0,i111]=th.min(img_map[i11,0,:,i111])
      max[i11,0,i111]=th.max(img_map[i11,0,:,i111])
      img_map[i11,0,:,i111]= (img_map[i11,0,:,i111]-min[i11,0,i111])/(max[i11,0,i111]-min[i11,0,i111]+0.0001)
    
    return  img_map  

def minmaxscaler(img_map):
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

lr1=0.01
def adjust_learning_rate(epoch):
    if epoch <= 30:  # 32k iterations
      lr=lr1
    elif epoch > 30 and epoch <= 50:  # 48k iterations
      lr=lr1/10
    elif epoch > 50 and epoch <= 62:  # 48k iterations
      lr=lr1/100
    elif epoch > 62 and epoch <= 70:
      lr=lr1/1000
    else:
      lr=lr1/1000
    return lr


log_dir='st-spcnncbam.pth.tar'

#log_file = open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/final.txt', 'a+')   
log_file6= open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/sp-stcnncbam/train_loss.txt', 'a+')


os.environ['CUDA_VISIBLE_DEVICES'] ='3'
def train(
        root='C:/Users/Admin/Desktop/HL/360video_our_all', #'E:/360_VRvideo','F:/360VRvideo', #"C:\Users\Admin\Desktop\HL\360_Saliency"
        bs=28,#25, #28
        lr=0.01,
        epochs=70,
        clear_cache=False,
        plot_server='http://127.0.0.1',
        plot_port=8097,
        save_interval=1,
        resume=True ,#True False
        start_epoch=0,
        exp_name='final',
        step_size=10,
        test_mode=True#False
):
   
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((112, 224)), #128, 256  #shuffle=true 数据集打乱 False    
        tf.ToTensor()
    ])
    dataset = VRVideo(root,112, 224, 20, frame_interval=1, cache_gt= True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    #print(dataset)  #128, 256,
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle= True  , num_workers=0, pin_memory=False )#True 
    print(type(loader))
    
    th.backends.cudnn.enabled = True
    model_cnn=Cnn()
    #model_cnnmap=Cnn_map()
    model_flo=flownet()
    #model_flomap=flo_map()
    model = Final11()  
    
    model_cnn= nn.DataParallel(model_cnn).cuda()
    #model_cnnmap=nn.DataParallel(model_cnnmap).cuda()
    model_flo=nn.DataParallel(model_flo).cuda()
    #model_flomap=nn.DataParallel(model_flomap).cuda()
    model =nn.DataParallel(model).cuda()
    
    optimizer_cnn = optim.SGD(model_cnn.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #optimizer_cnnmap = optim.SGD(model_cnnmap.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    optimizer_flo = optim.SGD(model_flo.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #optimizer_flomap = optim.SGD(model_flomap.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)  #优化器
    
    criterion = SphereMSE(56, 112).float().cuda() #+SFLoss(56, 112).float().cuda() #损失函数128,256
  
    if os.path.exists(log_dir):
        checkpoint = th.load(log_dir)
        model_cnn.load_state_dict(checkpoint['model_cnn_state_dict'])
        model_flo.load_state_dict(checkpoint['model_flo_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_cnn.load_state_dict(checkpoint['optimizer_cnn'])
        optimizer_flo.load_state_dict(checkpoint['optimizer_flo'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        checkpoint1=th.load('st-spcnn.pth.tar')
        model_cnn.load_state_dict(checkpoint1['model_cnn_state_dict'])
        model_flo.load_state_dict(checkpoint1['model_flo_state_dict'])
        #model.load_state_dict(checkpoint['model_state_dict'])
        
        print('无保存模型，将从头开始训练！')
        
        
    
    data_loss=[]
  # 
    for epoch in trange(start_epoch, epochs, desc='epoch'):
        train_loss=0.
        tic = time.time()
        lr=adjust_learning_rate(epoch)
        optimizer_cnn = optim.SGD(model_cnn.parameters(), lr, momentum=0.9, weight_decay=1e-5)
        optimizer_flo = optim.SGD(model_flo.parameters(), lr, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)  #优化器
  
        for i, (img1_batch,img2_batch , target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):#last_batch
        
            img1_var = Variable(img1_batch).cuda()
            img2_var = Variable(img2_batch).cuda()
            t_var = Variable(target_batch*10).cuda()
            data_time = time.time() - tic
            tic = time.time()
            
            optimizer_cnn.zero_grad()
            optimizer_flo.zero_grad()
            optimizer.zero_grad()
            
            out_cnn    = model_cnn(img1_var)
            out_flo    = model_flo(img1_var,img2_var) #, ,last_var ,img2_var
            out        = model(out_cnn, out_flo)
            #out1=th.clamp(out.data.cpu(), 3.0228e-04, 1, out=None)  
            
            loss = criterion(out, t_var)
       #     fwd_time = time.time() - tic
            tic = time.time()
            
            loss.backward()
            
            optimizer_cnn.step()
            #optimizer_cnnmap.step()
            optimizer_flo.step()
            #optimizer_flomap.step()
            optimizer.step()
            
            train_loss += loss.data[0]#*bs 
            
            
           # bkw_time = time.time() - tic
           
            msg1='{:d}, {:d} ,Train Loss: {:.6f} , loss: {:.7f}'.format(epoch,i,train_loss,loss.data[0] )
            #msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
             #   epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0])
            viz.images(img1_batch.cpu(), win='gt')  # * 10 abs(out.data.cpu().numpy()) np.maximum
           #viz.images(img2_batch.cpu(), win='gt2') #out_cnnmap.data
            viz.images( minmaxscaler(target_batch.cpu()), win='target') 
            viz.images( minmaxscaler(out.data.cpu()), win='out') #out.data.cpu().numpy()*1000
           
            #if (i+1) % 50 == 0:
             #  print(msg1, file=log_file, flush=True)
            #print(msg, flush=True)

            tic = time.time()
            del img1_batch,img1_var,img2_var,t_var, out #,out_cnn,out_cnnmap,out_flo,out_flomap
            th.cuda.empty_cache() 

        if (epoch+1 ) % save_interval == 0:#d and  train_loss != nan:
          
            th.save({'epoch': epoch, 
                          'model_cnn_state_dict': model_cnn.state_dict(),
                          'model_flo_state_dict': model_flo.state_dict(),
                          'model_state_dict': model.state_dict(),
                          'optimizer_cnn': optimizer_cnn.state_dict(),
                          'optimizer_flo': optimizer_flo.state_dict(),
                          'optimizer': optimizer.state_dict()}, 'st-spcnncbam.pth.tar')
       
        data_loss.append(train_loss)
        train_epoch_loss='{:d}   {:.6f}'.format(epoch,train_loss)
        print(train_epoch_loss,file=log_file6,flush=True)
                    
if __name__ == '__main__':
    
    Fire(train)
  