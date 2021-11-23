# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:51:29 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:20:04 2020

@author: Administrator
"""
import  torch
import visdom
import numpy as np
import torch as th
from sconv.module import SphericalConv#, SphereMSE
from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid
from gru  import ConvLSTMCell
from GRUcell  import ConvGRU 


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        #self.inplanes=16
        self.conv1 = SphericalConv(3, 16, np.pi/32, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None)
        self.bn1   = nn.BatchNorm2d(16)     #数据的归一化处理,对tensor求均值和方差
        self.relu1 = nn.ReLU(inplace=True)  #非线性激励函数
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = SphericalConv(16, 32, np.pi/16, kernel_size=(3, 3),stride=(1, 1), kernel_sr=None)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = SphericalConv(32, 64, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn3   = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = SphericalConv(64, 128, np.pi/2,  kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn4   = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)  #[14,7]
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = SphericalConv(128, 256, np.pi, kernel_size=(3, 3), stride=(1, 1),  kernel_sr=None)
        self.bn5   = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
    
        
        self.conv7 = SphericalConv(256+128, 256, np.pi/2, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn7   = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.Up7 = nn.Upsample(scale_factor=2) #[14,28]
        
        self.conv8 = SphericalConv(256+64, 128, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn8   = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU(inplace=True)
        self.Up8 = nn.Upsample(scale_factor=2)
        
        self.conv9 = SphericalConv(128+32, 64, np.pi/8, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn9   = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU(inplace=True)
        
      
    def forward(self, image):
        
        c1 = self.conv1(image) #th.cat([image, last], dim=1
        b1 = self.bn1(c1)
        r1 = self.relu1(b1)
        
        #x1 = self.ca(r1) * r1
        #y1 = self.sa(x1) * x1
         
        p1 = self.pool1(r1)
              

        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        r2 = self.relu2(b2)  #非线性激活层即保留大于0的值，即保留特征比较好的值，将特征小于0的值舍去
        p2 = self.pool2(r2)

        c3 = self.conv3(p2)
        b3 = self.bn3(c3)
        r3 = self.relu3(b3)
        p3 = self.pool3(r3)
        
        c4 = self.conv4(p3)
        b4 = self.bn4(c4)
        r4 = self.relu4(b4)
        p4 = self.pool4(r4)

        c5 = self.conv5(p4)
        b5 = self.bn5(c5)
        r5 = self.relu5(b5)
        
        c7 = self.conv7(th.cat([r5,p4], dim=1))
        b7 = self.bn7(c7)
        r7 = self.relu7(b7)
        d7 = self.Up7(r7)
        
        c8 = self.conv8(th.cat([d7,p3], dim=1))
        b8 = self.bn8(c8)
        r8 = self.relu8(b8)
        d8 = self.Up7(r8)
        
        c9 = self.conv9(th.cat([d8,p2], dim=1))  #torch.Size([2, 512, 3, 7])
        b9 = self.bn9(c9)
        r9 = self.relu9(b9)
        #print(c8.size())
         
        return r9
    
#net1= Cnn()
#print(net1)   
   
class flownet(nn.Module):
    def __init__(self):
        super(flownet, self).__init__()
        self.conv1 = SphericalConv(6, 32, np.pi/32, kernel_size=(7, 7), stride=(2, 2), kernel_sr=None)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = SphericalConv(32, 64, np.pi/16, kernel_size=(5, 5), stride=(2, 2), kernel_sr=None)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = SphericalConv(64, 128, np.pi/4, kernel_size=(5, 5), stride=(2, 2), kernel_sr=None)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv3_1 =SphericalConv(128, 128, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn3_1   = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU(inplace=True) #第四个卷积块  3*3*256 
        
        self.conv4 = SphericalConv(128, 256, np.pi/2, kernel_size=(3, 3), stride=(2, 2), kernel_sr=None )
        self.bn4   = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv4_1 = SphericalConv(256, 256, np.pi/2, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn4_1   = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU(inplace=True)   
        self.Up4_1 = nn.Upsample(scale_factor=2) #14,28 size=(14,28)
        
        self.conv5 = SphericalConv(256+128, 256, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn5   = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.Up5 = nn.Upsample(scale_factor=2)
        
        self.conv6 = SphericalConv(256+64, 128, np.pi/8, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn6   = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace=True)
      
        
    def forward(self,input ): #x1, x2
        
        #input = th.cat([x1, x2], dim=1)
        a1 =  self.conv1(input)
        h1 =  self.bn1(a1)
        w1 =  self.relu1(h1)
         
        a2 =  self.conv2(w1)
        h2 =  self.bn2(a2)
        w2 =  self.relu2(h2)
        
        
        a3 =  self.conv3(w2)     #a3.size[4, 256, 28, 56]
        h3 =  self.bn3(a3)
        w3 =  self.relu3(h3)
      
        
        a3_1 =  self.conv3_1(w3)  #a3_1.size[4, 256, 28, 56]
        h3_1 =  self.bn3_1(a3_1)
        w3_1 =  self.relu3_1(h3_1)
     
        
        a4 =  self.conv4(w3_1)  #a4.size[4, 256, 14, 28]
        h4 =  self.bn4(a4)
        w4 =  self.relu4(h4)
     
        
        a4_1 =  self.conv4_1(w4)   #a4_1.size[4, 256, 14, 28]
        h4_1 =  self.bn4_1(a4_1)
        w4_1 =  self.relu4_1(h4_1)
        d4_1 =  self.Up4_1(w4_1)
        
      
        
        a5 =  self.conv5(th.cat([d4_1, w3_1 ], dim=1))   #a5.size[4, 256, 7, 14]
        h5 =  self.bn5(a5)
        w5 =  self.relu5(h5)
        d5 =  self.Up5(w5)
        
        a6 =  self.conv6(th.cat([d5, w2 ], dim=1))   #a5.size[4, 256, 7, 14]
        h6 =  self.bn6(a6)
        w6 =  self.relu6(h6)
        
      
        
        return   w6 #concat_out
  



    

class Cnn_map(nn.Module):
    def __init__(self):
        super( Cnn_map, self).__init__()
       
        self.conv11 = SphericalConv(64, 1, np.pi/4, kernel_size=(3,3), kernel_sr=None, stride=(1,1), bias=False) #414 414
        self.up1 = nn.Upsample(scale_factor=2)#,mode='bilinear'
       
    def forward(self, feature):
      
       c11 = self.conv11(feature)
       c12 = self.up1(c11)
      
       return  c12   
        
class flo_map(nn.Module):
    def __init__(self):
        super(flo_map, self).__init__()
       
        self.conv11 = SphericalConv(128, 1, np.pi/4, kernel_size=(3,3), kernel_sr=None, stride=(1,1), bias=False) #414 414
        self.up1 = nn.Upsample(scale_factor=2)#,mode='bilinear'
       
    def forward(self, feature):
      
       c11 = self.conv11(feature)
       c12 = self.up1(c11)
      
       return  c12  
 
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 =nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
       
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
       
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = SphericalConv(2, 1,np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None, bias=False )
        #nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class  Final(nn.Module):   
     def __init__(self):
        super(Final, self).__init__()
        
        self.lastconv1 =  SphericalConv(128+64, 64,np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.Up1 = nn.Upsample(scale_factor=2)
        self.lastconv2 =  SphericalConv(64, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
        
        #self.Up2 = nn.Upsample(scale_factor=2)
      
        #self.lastconv4 =  SphericalConv(256, 128, np.pi/2, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
     def forward(self, cat1,cat2):#image1, image2
        #model2=flownet()
        #model2=model2#.cuda() #
        #self.concat_out = model2(image1,image2)  #th.cat([image1, image2], 1)
         
        MyFeature = th.cat([cat1,cat2], 1)  
        
        lastconv1 =  self.lastconv1(MyFeature)
       
        lastup1   =  self.Up1(lastconv1)
        lastconv2 =  self.lastconv2(lastup1)
     
       
        return  lastconv2
            
    
class  Final11(nn.Module):   
     def __init__(self):
        super(Final11, self).__init__()
        self.inplanes=64
        self.lastconv1 =  SphericalConv(128+64, 64,np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.ca = ChannelAttention1(self.inplanes)
        self.sa = SpatialAttention1()
        
        self.ca=nn.DataParallel(self.ca).cuda()
        self.sa=nn.DataParallel(self.sa).cuda()
        
        self.lastconv2 =  SphericalConv(128, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
        #self.bn2   = nn.BatchNorm2d(1)
        #self.relu2 = nn.ReLU(inplace=True)
        self.Up1 = nn.Upsample(scale_factor=2)
        
        #self.lastconv3 =  SphericalConv(64, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
        #self.bn3   = nn.BatchNorm2d(1)
        #self.relu3 = nn.ReLU(inplace=True)
        
     def forward(self, MyFeature):#cat1,cat2
       
      
        #MyFeature = th.cat([cat1,cat2], 1)  
        lastconv1 =  self.lastconv1(MyFeature)
        b1 = self.bn1( lastconv1)
        r1 = self.relu1(b1)
        MyFeature1=self.ca(r1)*  r1
        MyFeature1=self.sa(MyFeature1)* MyFeature1
        #[28,64,28,56]
        
        MyFeature2 = th.cat([r1,MyFeature1], 1)
        
        lastconv2 =  self.lastconv2(MyFeature2)
        #b2 = self.bn2( lastconv2)
        #r2 = self.relu2(b2)
        lastup1   =  self.Up1(lastconv2)
        
        #lastconv3 =  self.lastconv3(lastup1)
        #b3 = self.bn3( lastconv3)
        #r3 = self.relu3(b3)
      
       
        return lastup1
       
class  Final21(nn.Module):   
     def __init__(self):
        super(Final21, self).__init__()
        self.inplanes=64
        self.lastconv1 =  SphericalConv(128+64, 64,np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        
        self.ca=nn.DataParallel(self.ca).cuda()
        self.sa=nn.DataParallel(self.sa).cuda()
        
        self.lastconv2 =  SphericalConv(128, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
        #self.bn2   = nn.BatchNorm2d(1)
        #self.relu2 = nn.ReLU(inplace=True)
        self.Up1 = nn.Upsample(scale_factor=2)
        
        #self.lastconv3 =  SphericalConv(64, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
        #self.bn3   = nn.BatchNorm2d(1)
        #self.relu3 = nn.ReLU(inplace=True)
        
     def forward(self, cat1,cat2):#image1, image2
       
      
        MyFeature = th.cat([cat1,cat2], 1)  
        lastconv1 =  self.lastconv1(MyFeature)
        b1 = self.bn1( lastconv1)
        r1 = self.relu1(b1)
        MyFeature1=self.ca(r1)*  r1
        MyFeature1=self.sa(MyFeature1)* MyFeature1
        #[28,64,28,56]
        
        MyFeature2 = th.cat([r1,MyFeature1], 1)
        
        lastconv2 =  self.lastconv2(MyFeature2)
        #b2 = self.bn2( lastconv2)
        #r2 = self.relu2(b2)
        lastup1   =  self.Up1(lastconv2)
       
        return lastup1
    
    
    
    
    
#class  Final1(nn.Module):   
#     def __init__(self):
#        super(Final1, self).__init__()
        
#       self.lastconv1 =  SphericalConv(128+64, 64,np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )
#        self.bn1   = nn.BatchNorm2d(64)
#        self.relu1 = nn.ReLU(inplace=True)
#        self.Up1 = nn.Upsample(scale_factor=2)
#        self.lastconv2 =  SphericalConv(64, 1, np.pi/4, kernel_size=(3, 3), stride=(1, 1), kernel_sr=None )  
#        self.bn2   = nn.BatchNorm2d(1)
#        self.relu2 = nn.ReLU(inplace=True)
        
 #    def forward(self, cat1,cat2):#image1, image2
        
      
        # MyFeature = th.cat([cat1,cat2], 1)  
        
        # lastconv1 =  self.lastconv1(MyFeature)
        # b1 = self.bn1( lastconv1)
        # r1 = self.relu1(b1)
        # lastup1   =  self.Up1(r1)
        # lastconv2 =  self.lastconv2(lastup1)
        # b2 = self.bn2( lastconv2)
        # r2 = self.relu2(b2)  
    
        # return  r2   
    
    
# class salmap(nn.Module):
    # def __init__(self):
        # super(salmap, self).__init__()
        
        # self.deconv1 = nn.ConvTranspose2d(128,64, kernel_size=4,stride=2)
        # self.deconv2 = nn.ConvTranspose2d(64,1, kernel_size=4, stride=2)
        #self.deconv3 = nn.ConvTranspose2d(16,1, kernel_size=4, stride=2)
        #self.bn13   = nn.BatchNorm2d(1)
        #self.relu13 = nn.ReLU(inplace=True)
    # def forward(self, image1,image2):  #image1,image2
    
        # model4=Final()
        # last_feature=model4(image1,image2)
        # g1 = self.deconv1(last_feature)
        # g2 = self.deconv2(g1)
        
        # return g2
        
    
    
    
    
# class saliency_map(nn.Module):
    # def __init__(self):
        # super(saliency_map, self).__init__()
        
        # self.deconv1 = nn.ConvTranspose2d(128,64, kernel_size=4,stride=2)
        # self.deconv2 = nn.ConvTranspose2d(64,1, kernel_size=4, stride=2)
        #self.deconv3 = nn.ConvTranspose2d(16,1, kernel_size=4, stride=2)
        #self.bn13   = nn.BatchNorm2d(1)
        #self.relu13 = nn.ReLU(inplace=True)
    # def forward(self, last_feature): #image1,image2
       #model4=Final()
       #model3=model3#.cuda()
       #last_feature=model4(image1,image2) 
       #use_gpu = torch.cuda.is_available()
       #if use_gpu:
       #    dtype = torch.cuda.FloatTensor # computation in GPU
       #else:
       # dtype = torch.FloatTensor

       # height =14
       # width = 28
       # channels = 128
       # hidden_dim = [128, 128]
       # kernel_size = (3,3) # kernel size for two stacked hidden layer
       # num_layers = 2 # number of stacked hidden layer 
       # model1111 = ConvGRU(input_size=(height, width),
                    # input_dim=channels,
                    # hidden_dim=hidden_dim,
                    # kernel_size=kernel_size,
                    # num_layers=num_layers,
                    # dtype=dtype,
                    # batch_first=True,
                    # bias = True,
                    # return_all_layers = False)
       
       # last_feature=th.unsqueeze(last_feature,1)
        
       # layer_output_list, last_state_list =  model1111( last_feature ) 
       # layer_output = layer_output_list[0]
       # g1 = self.deconv1(layer_output[:,0,:,:,:])
       # g2 = self.deconv2(g1)
      # g3 = self.deconv3(g2)
       #g31= self.bn13(g3)
       #g33= self.relu13(g3)
       #print(g1.size())
       #print(g2.size())
       #print(g3.size())
       # return  g2       

        

 
   

    
    