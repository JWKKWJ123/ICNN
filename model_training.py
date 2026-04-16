import math
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import einops
import glob
import time
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc,balanced_accuracy_score
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import merge_ebms
from GL_ICNN import GL_ICNN





#define the early stop for the two traing stages
class EarlyStopping:
   
    def __init__(self, patience=30, verbose=False, delta=0, path='result/.pth', trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.bestmodel = None
        self.best = False
        self.early_stop_ebm = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.best = False
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            if self.counter >= self.patience*2:
                self.early_stop_ebm = True
                self.early_stop = True
        else:
            self.best = True
            self.best_score = score
            self.bestmodel = model
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
        
        
        
#save the ACC/AUC curve of the GL-ICNN (differnt traing stage in different colors), including both training set and validation set
def showlogs(logs,ki,path,trainstage):
        logs = np.array(torch.tensor(logs, device='cpu'))
        plt.plot( )
        plt.plot(logs[:, 1],color='blue')
        plt.plot(logs[:, 2],color='coral')
        plt.plot(logs[:trainstage, 1],color='cyan')
        plt.plot(logs[:trainstage, 2],color='orange')
        plt.legend(["train_GL-ICNN","val_GL-ICNN","train_GL-CNN","val_GL-CNN"],loc="lower right")
          
        plt.title("model AUC")
        plt.ylabel("AUC score")
        plt.xlabel("epoch")
        
        plt.tight_layout()
        plt.savefig(path.format(ki))
        


#save the loss curve of the validation set
def showloss(logs,ki,trainstage):
        logs = np.array(torch.tensor(logs, device='cpu'))
        plt.plot( )
        plt.plot(logs[:, 1],color='blue')
        #plt.plot(logs[:, 2],color='coral')
        plt.plot(logs[:trainstage, 1],color='cyan')
        #plt.plot(logs[:trainstage, 2],color='orange')
        plt.legend(["val_GL-ICNN","val_GL-ICNN"],loc="upper right")
        plt.tight_layout()
        plt.savefig('plot/_{}.png'.format(ki))
        

'''
dataloader:
In this binary classification case, images of diiferent class were put into different subfolders
The whole dataset was split into traing set, validation set and testing set 
The model is trained in 5-fold validation in this case
''' 
class SelfDataset(Dataset):
    def __init__(self, data_dir,ki=0, K=5, typ='train'):
        self.images = []
        self.labels = []
        self.names = []
        self.subnames = []


        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images)) 
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
            self.subnames += [os.path.relpath(imgs, data_dir)[2:-4] for imgs in images]
            
            
        ss1=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state= 0)
        
        
        train_index, test_index = ss1.split(self.images, self.labels)
        
        test_index = train_index[1]
        train_index = train_index[0]
        
        self.images, X_test = np.array(self.images)[train_index], np.array(self.images)[test_index]
        self.labels, y_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]
        self.subnames, name_test = np.array(self.subnames)[train_index], np.array(self.subnames)[test_index]
        
        if typ == 'test':
         self.crossimages = X_test
         self.crosslabels = y_test
         self.crossnames = name_test
           
        
        sfolder = StratifiedKFold(n_splits=K,random_state=0,shuffle=True)
        i=0
        for train, test in sfolder.split(self.images,self.labels):
            
            if i==ki:
               if typ == 'val':
            
            
                  self.crossimages = np.array(self.images)[test]
                  self.crosslabels = np.array(self.labels)[test]
                  self.crossnames = np.array(self.subnames)[test]
                  name_list = pd.DataFrame(self.crosslabels) 
                  #name_list.to_csv("validate.csv") 
                  
               elif typ == 'train':
                  self.crossimages = np.array(self.images)[train]
                  self.crosslabels = np.array(self.labels)[train]
                  self.crossnames = np.array(self.subnames)[train]
                  name_list = pd.DataFrame(self.crosslabels)
                  #name_list.to_csv("train.csv") 
                  
            i=i+1
        
    def __getitem__(self, idx):
        image = np.load(self.crossimages[idx])
        image = image[15:165,20:200,15:165]
        image = (image - np.mean(image))/ np.std(image)
        label = self.crosslabels[idx]
        name = self.crossnames[idx]
        return image, label, name
    def __getname__(self):
        return self.crossnames
    def __len__(self):
        return len(self.crossimages)
        
        
   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#define hyper-parameters
batchs = 8
K=5
epoch = 150
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
loss_fn = loss_fn.to(device)
# optimizer
learning_rate = 0.0005


#the file path of image data
root = '/data/ '
#initialize EBM
label = pd.read_csv(" ")
data = pd.read_csv(" )
ebm = ExplainableBoostingClassifier(interactions = 0)
ebm.fit(X,Y)


#the output file path, including plots, models, and tabular biomarkers
AUCcurvepath ='plot/.png'
modelpath ='result/.pth'
csvpath_train = 'biomarker/.csv'
csvpath_testsplit = 'biomarker/.csv'
label_train = 'biomarker/.csv'
label_test = 'biomarker/.csv'
ebmaucpath = 'biomarker/.csv'
changeepoch = 0




for ki in range(K):
    trainset = SelfDataset(root, ki=ki, typ='train')
    testset = SelfDataset(root, ki=ki, typ='val')
    testsetsplit = SelfDataset(root, ki=ki, typ='test')
    
    train_data_size = len(trainset)
    test_data_size = len(testset)
    change = 0
	  
    train_dataloader = DataLoader(
         dataset=trainset,
         batch_size=batchs,
         shuffle=True)
    test_dataloader = DataLoader(
         dataset=testset,
         batch_size=batchs,
     )
     
    test_dataloader_split = DataLoader(
         dataset=testsetsplit,
         batch_size=batchs,
     )  
    early_stopping = EarlyStopping(patience=30, path=modelpath.format(ki))
    
    #initialize GL-ICNN
    model = GL_ICNN(1,
                        W =150,
                        D =180,
                        H =150
                        
                        )
    
    model = model.to(device)
    optimzer = torch.optim.Adam(params= model.parameters(), lr = learning_rate)
    # num of training steps
    total_train_step = 0
    # num of testing steps
    total_test_step = 0
    logs = []
    loss_list = []
    ebm_auc = []
    max_acc = 0.5
    #start the traing of a fold
    for i in range(epoch):
     print("--------start training epoch: {}-------------------".format(i + 1))

     # start of training
     model.train()
     total_test_loss = 0
     total_trainaccuracy = 0
     total_testaccuracy = 0
     subject_auc = []

     if early_stopping.early_stop==0:
     #start the training of the first stage 
      output = torch.empty(0,1).to(device)
      label = torch.empty(0,1).to(device)
      for data in train_dataloader:
        imgs, targets ,names = data
        imgs = imgs.unsqueeze(1)
        imgs = imgs.to(torch.float32)
        targets = targets.unsqueeze(1)
        targets = targets.to(torch.float32)
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs, outputs_ebm, biomarkers = model(imgs,ebm)
        outputs_ebm = outputs_ebm.to(device)
        outputs = outputs.to(device)
        accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()
        total_trainaccuracy = total_trainaccuracy + accuracy
        print('accuracy:',accuracy/batchs)
        predicts = torch.where(outputs > 0.5, 1, 0)
        loss = loss_fn(outputs, targets)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        output = torch.cat([output,outputs],0)
        label = torch.cat([label,targets],0)

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("num of training:{},loss:{}".format(total_train_step, loss.item()))
            
            
      output_pro = output.detach().cpu().numpy()
      label = label.detach().cpu().numpy()
      auc_train=roc_auc_score(label,output_pro)
      #start the validation of each epoch
      model.eval()
      output = []
      label = []
      output = torch.empty(0,1).to(device)
      label = torch.empty(0,1).to(device)

      with torch.no_grad():
        for data in test_dataloader:
            imgs, targets,names = data
            imgs = imgs.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs,outputs_ebm,biomarkers = model(imgs,ebm)
            outputs = outputs.to(device)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()
            total_testaccuracy = total_testaccuracy + accuracy
            output = torch.cat([output,outputs],0)
            label = torch.cat([label,targets],0)
       
      output_pro = output.cpu().numpy()
      output = np.where(output_pro > 0.5, 1, 0)
      label = label.cpu().numpy()
      ba = balanced_accuracy_score(label,output)
      auc_val=roc_auc_score(label,output_pro)
      #In this case, early stop is depend on balanced accuracy
      early_stopping(100-ba, model)

      if early_stopping.early_stop:
         model = early_stopping.bestmodel

      print("train accuracy: {}".format(total_trainaccuracy / train_data_size))
      print("test Loss: {}".format(total_test_loss))
      print("test accuracy: {}".format(total_testaccuracy / test_data_size))
      logs.append([epoch, auc_train, auc_val])
      loss_list.append([epoch,total_test_loss])
      
      total_test_step += 1
      
      showloss(loss_list,ki,trainstage=0)
      showlogs(logs,ki,AUCcurvepath, trainstage=0)
      plt.close()
         


     #start the training of the second stage 
     if early_stopping.early_stop_ebm==0 and early_stopping.early_stop==1:
      change += 1
      if change ==1:
         changeepoch = i
      total_test_loss = 0
      total_trainaccuracy = 0
      total_testaccuracy = 0
      subject_auc = []
      learning_rate = 0.001
      output = []
      label = []
      output = torch.empty(0,1).to(device)
      label = torch.empty(0,1).to(device)
      j = 0
      with torch.no_grad():
       for data in train_dataloader:
           
            imgs, targets,names = data
            imgs = imgs.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            outputs,outputs_ebm,biomarkers = model(imgs,ebm)
            
            biomarkers = biomarkers.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            
            if j == 0:
              biomarker_list = biomarkers
              target_list = targets
              
            else:
              biomarker_list = np.concatenate((biomarker_list, biomarkers),axis = 0)
              target_list = np.concatenate((target_list, targets),axis = 0)
            j = j+1
              
            print(biomarker_list.shape)
            print(target_list.shape)
            
      unmerge_ebm = ExplainableBoostingClassifier(interactions = 0)
      unmerge_ebm.fit(biomarker_list,target_list)
        
      biomarker_list_train = pd.DataFrame(biomarker_list)    
      target_list_train = pd.DataFrame(target_list)
      
      #For now we don't include the merge of EBM, so the EBM trained from scratch at every epoch 
      '''
      if change == 1:
           ebm = unmerge_ebm
      else:
           ebm = merge_ebms([ebm, unmerge_ebm])
      '''

      output_pro = ebm.predict_proba(biomarker_list)
      output_pro = output_pro[:,1]
      auc=roc_auc_score(target_list,output_pro)
      print('auc:',auc)
      
      output = torch.empty(0,1).to(device)
      label = torch.empty(0,1).to(device)
      model.train()
      for data in train_dataloader:
        imgs, targets ,names = data
        imgs = imgs.unsqueeze(1)
        imgs = imgs.to(torch.float32)
        targets = targets.unsqueeze(1)
        targets = targets.to(torch.float32)
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs, outputs_ebm, biomarkers = model(imgs,ebm)

        outputs_ebm = outputs_ebm.to(device)
        outputs = outputs.to(device)      
        accuracy = (torch.where(outputs_ebm > 0.5, 1, 0) == targets).sum()
       
        total_trainaccuracy = total_trainaccuracy + accuracy
        print('accuracy:',accuracy/batchs)
        predicts = torch.where(outputs > 0.5, 1, 0)
        loss = loss_fn(outputs_ebm, targets)
        output = torch.cat([output,outputs_ebm],0)
        label = torch.cat([label,targets],0)
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("num of training:{},loss:{}".format(total_train_step, loss.item()))
            
      output_pro = output.detach().cpu().numpy()
      label = label.detach().cpu().numpy()
      auc_train=roc_auc_score(label,output_pro)      
      #start the validation of each epoch
      model.eval()
      output = []
      label = []
      output = torch.empty(0,1).to(device)
      label = torch.empty(0,1).to(device)
      j=0

      with torch.no_grad():
        for data in test_dataloader:
            imgs, targets,names = data
            imgs = imgs.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs,outputs_ebm,biomarkers = model(imgs,ebm)
            outputs_ebm = outputs_ebm.to(device)
            loss = loss_fn(outputs_ebm, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (torch.where(outputs_ebm > 0.5, 1, 0) == targets).sum()
            total_testaccuracy = total_testaccuracy + accuracy
            output = torch.cat([output,outputs_ebm],0)
            label = torch.cat([label,targets],0)
            biomarkers = biomarkers.cpu().numpy()  
            targets = targets.cpu().numpy()  
            
            
            if j == 0:
              biomarker_list_test = biomarkers
              target_list_test = targets
              
            else:
              biomarker_list_test = np.concatenate((biomarker_list_test, biomarkers),axis = 0)
              target_list_test = np.concatenate((target_list_test, targets),axis = 0)
            j = j+1
            
      ebm_output_pro = unmerge_ebm.predict_proba(biomarker_list_test)
      ebm_output_pro = ebm_output_pro[:,1]
      auc=roc_auc_score(target_list_test,ebm_output_pro)
      subject_auc.append(auc)
      ebm_output_pro = ebm.predict_proba(biomarker_list_test)
      ebm_output_pro = ebm_output_pro[:,1]
      auc=roc_auc_score(target_list_test,ebm_output_pro)
      subject_auc.append(auc)
      ebm_auc.append(subject_auc)
      ebm_auc_csv = pd.DataFrame(ebm_auc)    
      ebm_auc_csv.to_csv(ebmaucpath.format(ki))
      
      
       
      output_pro = output.cpu().numpy()
      output = np.where(output_pro > 0.5, 1, 0)
      label = label.cpu().numpy()
      ba = balanced_accuracy_score(label,output)
      auc_val = roc_auc_score(label,output_pro)
      early_stopping(100-ba, model)

      print("train accuracy: {}".format(total_trainaccuracy / train_data_size))
      print("test Loss: {}".format(total_test_loss))
      print("test accuracy: {}".format(total_testaccuracy / test_data_size))
      logs.append([epoch, auc_train, auc_val])
      loss_list.append([epoch,total_test_loss])
      total_test_step += 1
      if early_stopping.best or (1-os.path.exists(csvpath_train.format(ki))):
               biomarker_list_train.to_csv(csvpath_train.format(ki),index=False)
               target_list_train.to_csv(label_train.format(ki))
               
     if early_stopping.early_stop_ebm:
         total_test_loss = 0
         model.eval()
         output = []
         label = []
         output = torch.empty(0,1).to(device)
         label = torch.empty(0,1).to(device)
         j=0
         
         
        
         print("Early stopping")
         #start the testing of the model
         with torch.no_grad():
            for data in test_dataloader_split:
              imgs, targets,names = data
              imgs = imgs.unsqueeze(1)
              imgs = imgs.to(torch.float32)
              targets = targets.unsqueeze(1)
              targets = targets.to(torch.float32)
              imgs = imgs.to(device)
              targets = targets.to(device)
              outputs,outputs_ebm,biomarkers = model(imgs,ebm)
              outputs_ebm = outputs_ebm.to(device)
              loss = loss_fn(outputs_ebm, targets)
              total_test_loss = total_test_loss + loss.item()
              accuracy = (torch.where(outputs_ebm > 0.5, 1, 0) == targets).sum()
              total_testaccuracy = total_testaccuracy + accuracy
              output = torch.cat([output,outputs_ebm],0)
              label = torch.cat([label,targets],0)
              biomarkers = biomarkers.cpu().numpy()  
              targets = targets.cpu().numpy()  
            
            
              if j == 0:
                biomarker_list_test = biomarkers
                target_list_test = targets
              
              else:
                biomarker_list_test = np.concatenate((biomarker_list_test, biomarkers),axis = 0)
                target_list_test = np.concatenate((target_list_test, targets),axis = 0)
              j = j+1
         
         
         output = output.cpu().numpy()
 
         ebm_output_pro = unmerge_ebm.predict_proba(biomarker_list_test)
         ebm_output_pro = ebm_output_pro[:,1]
         auc=roc_auc_score(target_list_test,ebm_output_pro)
         subject_auc.append(auc)
         ebm_output_pro = ebm.predict_proba(biomarker_list_test)
         ebm_output_pro = ebm_output_pro[:,1]
         auc=roc_auc_score(target_list_test,ebm_output_pro)
         
         auc_final = roc_auc_score(target_list_test,output)
         output = np.where(output > 0.5, 1, 0)  
         ba_final = balanced_accuracy_score(target_list_test,output)
         
         subject_auc.append(auc)
         subject_auc.append(ba_final)
         subject_auc.append(auc_final)
         ebm_auc.append(subject_auc)
         ebm_auc = pd.DataFrame(ebm_auc)    
         ebm_auc.to_csv(ebmaucpath.format(ki))
         biomarker_list_test = pd.DataFrame(biomarker_list_test)    
         biomarker_list_test.to_csv(csvpath_testsplit.format(ki),index=False)
         target_list_test = pd.DataFrame(target_list_test)
         target_list_test.to_csv(label_test.format(ki))
         break
      
     showloss(loss_list,ki,trainstage=changeepoch)
     plt.close()
     showlogs(logs,ki,AUCcurvepath,trainstage=changeepoch)
     plt.close()  

