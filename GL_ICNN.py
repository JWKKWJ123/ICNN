import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import copy
import math
import math
import pandas as pd
import numpy as np
import os







class dense_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(vgg_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(inplanes, 16 , kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes+16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        residual = self.conv3(residual)

        out = torch.cat((out,residual),axis=1)
        out = self.bn3(out)
        out = self.relu(out)
        return out






class vgg_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(vgg_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.conv1(x)
        
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.bn1(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        return out






#build the GLobal-CNN backbone
class GlobalCNN(nn.Module):
    def __init__(self):
        super(Global, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = vgg_block(16, 16)
        self.block2 = vgg_block(32, 32)
        self.block3 = vgg_block(48, 48)
        self.block4 = vgg_block(64, 64)
        self.conv_cls = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_cls(x)
        #x = torch.sigmoid_(x)
        return x
        


#build the Local-CNN backbone
class LocalCNN(nn.Module):
    def __init__(self):
        super(dyrbaVGG, self).__init__()

        self.block1 = vgg_block(1, 10)
        self.conv_cls = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64,10),
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.conv_cls(x)
        #x = torch.sigmoid_(x)
        return x





class EBM_layer(nn.Module):



    def __init__(self, **kwargs):
        super(EBM_layer, self).__init__(**kwargs)

    def forward(self, x, ebm):
    
        x = x.detach().cpu().numpy()
        print(x.shape)
        output_pro_ebm = ebm.predict_proba(x)
        output_pro_ebm = output_pro_ebm[:,1]
        output_pro_ebm = torch.tensor(output_pro_ebm, requires_grad=True)
        output_pro_ebm = output_pro_ebm.unsqueeze(1)
        return output_pro_ebm




#initialize EBM
label = pd.read_csv(" ")
data = pd.read_csv(" )
ebm = ExplainableBoostingClassifier(interactions = 0)
ebm.fit(X,Y)








class GL_ICNN(nn.Module):
    def __init__(self,inplace,
                 W=150,
                 D=180,
                 H=150,
                 drop_rate=0.2,
                 backbone=LocalCNN,ebm_backbone = ebm):
        """
        Parameter:
            @inplace: namber of channels (1)
            @(W,D,H): size of the input image, will determine the patch size and stride of image patches
            @step: the step size of the sliding window of the local patches
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """
        
        super().__init__()
        
        
        
        self.hidden_size = 10
        
        
        #generate patch size and stride based on the size of the input image, the patch size is equal to stride now
        stride_W = W // 3
        stride_H = H // 3
        stride_D = D // 3
        
        remainder_W = W % 3 // 2
        remainder_H = H % 3 // 2
        remainder_D = D % 3 // 2


        i=0
        self.cnnlist = nn.ModuleList()
        self.global_feat = GlobalCNN()
        
        names = self.__dict__
        for z in range(remainder_D,D-stride_D+1,stride_D):         
           for y in range(remainder_H,H-stride_H+1,stride_H):
              for x in range(remainder_W,W-stride_W+1,stride_W):
                i+=1
                self.cnnlist.append(backbone())

        self.avg = nn.AdaptiveAvgPool3d(1) 
        print(i*self.hidden_size)
        #build FC layers of the GL-CNN
        self.globallinear = nn.Linear(80,20)
        self.extralinear1 = nn.Linear(i*self.hidden_size+20,128)
        self.extralinear2 = nn.Linear(128,64)
        self.hybridout=nn.Linear(64,1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid( )
        self.ebm = ebm
        #EBM is the output block of the GL-ICNN
        self.EBM_layer = EBM_layer()
        
        
        
        
    def forward(self,xinput,ebm):
        B,C,W,D,H=xinput.size()
        outlist = []
        i=0
        out_all = self.global_feat(xinput)
        out_all=self.relu(out_all)
        out_all=self.dropout(out_all)
        out_all=self.globallinear(out_all)
        biomarkers = torch.mean(out_all,dim=1,keepdim=True)
        
                
        for z in range(remainder_D,D-stride_D+1,stride_D):         
           for y in range(remainder_H,H-stride_H+1,stride_H):
              for x in range(remainder_W,W-stride_W+1,stride_W):
                 locx = xinput[:,:,x:x+stride_W,z:z+stride_D,y:y+stride_H]
                 xloc = self.cnnlist[i](locx)
                 out =self.relu(xloc)
                 out =self.dropout(out)
                 loc_biomarkers = torch.mean(out,dim=1,keepdim=True)
                 biomarkers = torch.cat([biomarkers,loc_biomarkers],1)
                 out_all=torch.cat([out_all,out],1)
                 i=i+1
                
        print(out_all.shape[1],'output')
        out_all=self.relu(out_all)
        print(out_all.shape)
        dense=self.extralinear1(out_all)
        dense=self.relu(dense)
        dense=self.dropout(dense)
        dense=self.extralinear2(dense)
        dense=self.relu(dense)
        dense=self.dropout(dense)
        dense=self.hybridout(dense)
        
        prediction_CNN=self.sigmoid(dense)
        prediction_ICNN = self.EBM_layer(biomarkers,ebm)
                
        #output the results of GL-CNN and GL-ICNN, and the features of the CNN part (biomarkers)
        return prediction_CNN, prediction_ICNN, biomarkers
                
                

