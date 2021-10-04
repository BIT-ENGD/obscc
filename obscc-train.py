import torch 
from znprompt import znprompt as zp
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim 
import os
import shutil
from pathlib import Path 
from torchsummaryX import summary as summaryX
import onnx 
import pickle as pk 
from  traininfo import TrainInfo
import json 
import time
from  numba  import jit ,njit 

DATA_DIR="/data/ai/bit-engd/obscc/"

JSON_FILE="languages.json"

MODEL_NAME="src_cat.pth"
ONNX_MODEL_PATH="src_cat.onnx"

DATASET_FILE_X="data/ds_x_1994419.dat"
DATASET_FILE_Y="data/ds_y_1994419.dat"
CAT_FILE="data/allcat.dat"

MAX_TOKEN =1000

VOCAB_FILE_REAL="data/vocab_all.dat"
VOCAB_FILE_DICT="data/vocab_dict.dat"
BATCH_SIZE=32
FILTER_NUM=128
EMBED_DIM=128

DROPOUT=0.5
EPOCH_NUM=50
def save_var(varname,filename):
    with open(filename,"wb") as f:
        pk.dump(varname,f)

def load_var(filename):
    with open(filename,"rb") as f:
        return pk.load(f)


class textCNN_M(nn.Module):
    def __init__(self, vocab_size,Embedding_size,num_classs):
        super(textCNN_M, self).__init__()
             
        Vocab = vocab_size ## 已知词的数量
        Dim = Embedding_size##每个词向量长度
        Cla = num_classs##类别数
        Ci = 1 ##输入的channel数
        Knum = FILTER_NUM ## 每种卷积核的数量
        Ks = [2,3,5] ## 卷积核list，形如[2,3,4]
        
        self.embed = nn.Embedding(Vocab,Dim) ## 词向量，这里直接随机
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Knum,(K,Dim)) for K in Ks]) ## 卷积层
        self.dropout = nn.Dropout(DROPOUT) 
        self.fc = nn.Linear(len(Ks)*Knum,Cla) ##全连接层
        
    def forward(self,x):
        x = self.embed(x) #(N,W,D)
        
        x = x.unsqueeze(1) #(N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line,int(line.size(2))).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        
        x = torch.cat(x,1) #(N,Knum*len(Ks))
        
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


class ZnQmDataset(Dataset):
    def __init__(self,DATADIR,DATAFILE_X,DATAFILE_Y):
        super(ZnQmDataset,self).__init__()
                
        self.x_data=load_var(os.path.join(DATADIR,DATAFILE_X))
        self.y_data=load_var(os.path.join(DATADIR,DATAFILE_Y))
        self.x_data=torch.tensor(self.x_data)
        self.y_data=torch.tensor(self.y_data)
   
    def __len__(self):
        return len (self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    

def doTrain(ds,vocab_dict,all_cat):
    device      = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    loader      = DataLoader(dataset=ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=16)
    model       = textCNN_M(len(vocab_dict),EMBED_DIM,len(all_cat)).to(device)
    optimizer   = optim.Adam(model.parameters(),lr=5e-4)
    criterion   = nn.CrossEntropyLoss().to(device)
    TrainInfoObj=TrainInfo()
  
    for epoch in range(EPOCH_NUM):
        for batch_x,batch_y in loader:
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            pred=model(batch_x)
            loss=criterion(pred,batch_y)
            TrainInfoObj.add_scalar('loss', loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def doValid(ds,vocab_dict,all_cat):
    pass 

if __name__=="__main__":
    vocab_dict=load_var(os.path.join(DATA_DIR,VOCAB_FILE_DICT))
    all_cat=load_var(os.path.join(DATA_DIR,CAT_FILE))
    ds=ZnQmDataset(DATA_DIR,DATASET_FILE_X,DATASET_FILE_Y)

    doTrain(ds,vocab_dict,all_cat)