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
import copy
from  sp_net  import Final ,flownet, Cnn  ,Cnn_map,flo_map # spherical_unet
from sconv.module import SphereMSE #,SFLoss

#import  pdb
def minmaxscaler1(img_map):
    
    img_map=np.maximum(img_map, 0)
    min=th.zeros(img_map.size()[0],1, img_map.size()[2])
    max=th.zeros(img_map.size()[0],1,img_map.size()[2])
    for i11 in range(img_map.size()[0]):
     for i111 in range(img_map.size()[2]):
      min[i11,0,i111]=th.min(img_map[i11,0,i111,:])
      max[i11,0,i111]=th.max(img_map[i11,0,i111,:])
      img_map[i11,0,i111,:]= (img_map[i11,0,i111,:]-min[i11,0,i111])/(max[i11,0,i111]-min[i11,0,i111]+0.0001)
    
    return  img_map  

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

def Accuracy_sig(he_map):
    block_h=14
    block_w=14
    min=th.zeros(he_map.size()[0],1,1)
    max=th.zeros(he_map.size()[0],1,1)
    for i11 in range(he_map.size()[0]):
      min[i11]=th.min(he_map[i11])
      max[i11]=th.max(he_map[i11])
      he_map[i11]= (he_map[i11]-min[i11])/(max[i11]-min[i11]+0.00001)
      
    a_min=th.zeros(he_map.size()[0],1,int(56/block_h)*int(112/block_w))   
    he_map_sig=th.zeros(int(56/block_h)*int(112/block_w),he_map.size()[0])
    he_map_sig=he_map_sig.int()
    for i2 in range(he_map.size()[0]):
      blocks=1
      for i3 in range(int(56/block_h)):
         for i4 in range(int(112/block_w)):          
           a_min[i2,:,i3*4+i4]=(he_map[i2,:,i3*block_h:(i3+1)*block_h-1, i4*block_w:(i4+1)*block_w-1]).mean()
           if a_min[i2, 0,i3*4+i4] >= 0.2:
             he_map_sig[i3*4+i4,i2]=blocks
           blocks= blocks+1
    
    he_map_sig= th.transpose(he_map_sig,1,0) 
    return he_map_sig
   
def Accuracy_sig1(he_map):
    block_h=14
    block_w=14
    min=th.zeros(he_map.size()[0],1,1)
    max=th.zeros(he_map.size()[0],1,1)
    for i11 in range(he_map.size()[0]):
      min[i11]=th.min(he_map[i11])
      max[i11]=th.max(he_map[i11])
      he_map[i11]= (he_map[i11]-min[i11])/(max[i11]-min[i11]+0.00001)
    a_min=th.zeros(he_map.size()[0],1,int(56/block_h)*int(112/block_w))   
    he_map_sig=th.zeros(int(56/block_h)*int(112/block_w),he_map.size()[0])
    he_map_sig=he_map_sig.int()
    
    for i2 in range(he_map.size()[0]):
      blocks=1
      for i3 in range(int(56/block_h)):
         for i4 in range(int(112/block_w)):          
           a_min[i2,:,i3*4+i4]=(he_map[i2,:,i3*block_h:(i3+1)*block_h-1, i4*block_w:(i4+1)*block_w-1]).mean()
           if a_min[i2, 0,i3*4+i4] != 0:
             he_map_sig[i3*4+i4,i2]=blocks
           blocks= blocks+1
    #he_map_sig = th.gt(a_min,0.2)
             #he_map_sig[i2,0,i3*4+i4]==1
    he_map_sig= th.transpose(he_map_sig,1,0)
    return he_map_sig    

def dif(liA,liB):
    #求交集的两种方式
    #liA=liA .numpy().tolist()
    #liB= 
    count_num=0
    reta_sum=[]
    retb_sum=[]
    reta_p=th.zeros(liA.size()[0])
    reta_r=th.zeros(liB.size()[0])
    retA=[]
    for i111 in range(liA.size()[0]):
      
      #retA=  
      retA.append(list(set(liA[i111]).intersection(set(liB[i111]))))
      retA[i111] = th.from_numpy(np.array(retA[i111]))
      
      reta_sum.append(liA.size()[1]-liA[i111].numpy().tolist().count(count_num))
      
      retb_sum.append(liB.size()[1]-liB[i111].numpy().tolist().count(count_num))
      reta_p[i111]=((retA[i111].size()[0]-1)/ reta_sum[i111]+0.0001)
      reta_r[i111]=((retA[i111].size()[0]-1)/ retb_sum[i111]+0.0001)
      #reta_p[i111]= th.from_numpy(np.array(reta_p[i111]))
      #reta_r[i111]= th.from_numpy(np.array(reta_p[i111]))
      #retA = [j11 for j11 in liA[i111] if j11 in liB[i111]]
   # retB = list(set(listA).intersection(set(listB)))
     
    return retA,reta_p,reta_r

log_dir='model_final5_1_1.pth.tar'

log_file = open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/final.txt', 'a+')   
log_file1 = open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/out_sig.txt', 'a+')
log_file2= open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/g_sig.txt', 'a+')
log_file3= open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/jiao_sig.txt', 'a+')
log_file4= open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/out_accrreact1.txt', 'a+')
log_file5= open('C:/Users/Admin/Desktop/HL/Saliency-detection-in-360-video-master/saliency5/val_out.txt', 'a+')
log_file6= open('final_saliency_epoch_loss.txt', 'a+')

def evalute(model_cnn,model_flo, model,loader,epoch, criterion):
    model_cnn.eval()
    model_flo.eval()
    model.eval()
    test_loss=0.
    test_acc = 0.
    test_recall=0.
    total = len(loader)
    for i, (img1_batch,img2_batch , target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
        img1_var = Variable(img1_batch,requires_grad=False).cuda()
        img2_var = Variable(img2_batch,requires_grad=False).cuda()
        t_var = Variable(target_batch,requires_grad=False).cuda()
        out_cnn    = model_cnn(img1_var)
        out_flo    = model_flo(img1_var,img2_var) #, ,last_var ,img2_var
        out        = model(out_cnn, out_flo)
        out1=th.clamp(out.data.cpu(), 3.0228e-04, 1, out=None)  
            
        loss = criterion(out, t_var)
        test_loss += loss.data[0]*out.size()[0] 
        out_sig= Accuracy_sig(out1).data.cpu()
        t_var_sig=Accuracy_sig1(target_batch*25)
        out_jiao, out_acc,out_recall=dif(out_sig,t_var_sig)
        
        test_acc= test_acc+ th.sum(out_acc)/out.size()[0]
        test_recall=test_recall+ th.sum(out_recall)/out.size()[0]
        ms_sig='{:d}  {:d}  Acc: {:.4f}   recall: {:.4f}'.format(epoch,i,th.sum(out_acc)/out.size()[0] ,th.sum(out_recall)/out.size()[0]  )
        print(out_sig, file=log_file1, flush=True)
        print(t_var_sig, file=log_file2, flush=True)
        print( out_jiao, file=log_file3, flush=True)
        print(ms_sig, file=log_file4, flush=True) 
        #correct += torch.eq(pred, y).sum().float().item()
    return test_loss/total,test_acc/total,test_recall/total

os.environ['CUDA_VISIBLE_DEVICES'] ='1,2'
def train(
        root='C:/Users/Admin/Desktop/HL/360video_our_all', #'E:/360_VRvideo','F:/360VRvideo', #"C:\Users\Admin\Desktop\HL\360_Saliency"
        bs=25,#25, #28
        lr=0.002,#0.0002,
        epochs=183,
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
    # pynvml.nvmlInit()
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((112, 224)), #128, 256  #shuffle=true 数据集打乱 False    
        tf.ToTensor()
    ])
    dataset = VRVideo(root,112, 224, 15, frame_interval=1, cache_gt= True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    print(dataset)  #128, 256,
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle= True  , num_workers=16, pin_memory=False )#True 
    print(type(loader))
    
    th.backends.cudnn.enabled = True
    model_cnn=Cnn()
    model_cnnmap=Cnn_map()
    model_flo=flownet()
    model_flomap=flo_map()
    model = Final()  
    
    model_cnn= nn.DataParallel(model_cnn).cuda()
    model_cnnmap=nn.DataParallel(model_cnnmap).cuda()
    model_flo=nn.DataParallel(model_flo).cuda()
    model_flomap=nn.DataParallel(model_flomap).cuda()
    model =nn.DataParallel(model).cuda()
    
    #optimizer_cnn = optim.SGD(model_cnn.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #optimizer_cnnmap = optim.SGD(model_cnnmap.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #optimizer_flo = optim.SGD(model_flo.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #optimizer_flomap = optim.SGD(model_flomap.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)  #优化器
    #scheduler  = th.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    
    #scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, 
    #verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    criterion = SphereMSE(56, 112).float().cuda() #+SFLoss(56, 112).float().cuda() #损失函数128,256
    #if resume:
        
        #ckpt = th.load('model_final1.pth.tar' ) #'ckpt-' + exp_name +'-latest.pth.tar'
        #model.load_state_dict(ckpt['model_state_dict'])   #'''应用到网络结构中''''state_dict'
        #start_epoch = ckpt['epoch']
   
    if os.path.exists(log_dir):
        checkpoint = th.load(log_dir)
        model_cnn.load_state_dict(checkpoint['model_cnn_state_dict'])
        model_flo.load_state_dict(checkpoint['model_flo_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
        
        
    
    data_loss=[]
  # 
    for epoch in trange(start_epoch, epochs, desc='epoch'):
        train_loss=0.
        tic = time.time()
        #scheduler.step()  
        for i, (img1_batch,img2_batch , target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):#last_batch
        
            img1_var = Variable(img1_batch).cuda()
            img2_var = Variable(img2_batch).cuda()
            t_var = Variable(target_batch*10).cuda()
            data_time = time.time() - tic
            tic = time.time()
            
            optimizer.zero_grad()
            out_cnn    = model_cnn(img1_var)
            out_cnnmap = model_cnnmap(out_cnn)
            out_flo    = model_flo(img1_var,img2_var) #, ,last_var ,img2_var
            out_flomap = model_flomap(out_flo)
            out        = model(out_cnn, out_flo)
            out1=th.clamp(out.data.cpu(), 3.0228e-04, 1, out=None)  
            
            loss = criterion(out, t_var)
            fwd_time = time.time() - tic
            tic = time.time()
            
            loss.backward()
            
            #optimizer_cnn.step()
            #optimizer_cnnmap.step()
            #optimizer_flo.step()
            #optimizer_flomap.step()
            optimizer.step()
            
            train_loss += loss.data[0]#*bs 
            
            
            bkw_time = time.time() - tic
           
            msg1='{:d}, {:d} ,Train Loss: {:.6f} , loss: {:.7f}'.format(epoch,i,train_loss,loss.data[0] )
            msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
                epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0])
            viz.images(img1_batch.cpu(), win='gt')  # * 10 abs(out.data.cpu().numpy()) np.maximum
            viz.images(minmaxscaler(out1), win='cnn1') #out_cnnmap.data
            viz.images( minmaxscaler(target_batch.cpu()), win='target') 
            #viz.images( minmaxscaler1(out.data.cpu()), win='out') #out.data.cpu().numpy()*1000
            viz.images( minmaxscaler1(th.clamp(out_cnnmap.data.cpu(), 3.0228e-04, 1, out=None)), win='out_cnnmap') 
            viz.images( minmaxscaler(th.clamp(out_flomap.data.cpu(), 3.0228e-04, 1, out=None)), win='out_flomap') 
            ##viz.images((out*100).data.cpu(), win='flo12')
             #viz.text(msg, win='log')
           
            if (i+1) % 50 == 0:
              print(msg1, file=log_file, flush=True)
            #print(msg, flush=True)

            tic = time.time()
            del img1_batch,img1_var,img2_var,t_var, out #,out_cnn,out_cnnmap,out_flo,out_flomap
            th.cuda.empty_cache() 

        if (epoch+1 ) % save_interval == 0:#d and  train_loss != nan:
          
            th.save({'epoch': epoch, 
                          'model_cnn_state_dict': model_cnn.state_dict(),
                          'model_flo_state_dict': model_flo.state_dict(),
                          'model_state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}, 'model_final5_1_1.pth.tar')
       
            th.save({'epoch': epoch, 
                          'model_cnnmap_state_dict': model_cnnmap.state_dict(),
                          'model_flomap_state_dict': model_flomap.state_dict()},
                    'model_map5_1_1.pth.tar') 
            
        data_loss.append(train_loss)
        train_epoch_loss='{:d}   {:.6f}'.format(epoch,train_loss)
        print(train_epoch_loss,file=log_file6,flush=True)
                    
if __name__ == '__main__':
    
    Fire(train)
  