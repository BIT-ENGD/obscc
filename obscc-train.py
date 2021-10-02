import torch 
from znprompt import znprompt as zp
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
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

DATASET_FILE="ds.dat"
CAT_FILE="allcat.dat"

MAX_TOKEN =1000

VOCAB_FILE_REAL="data/vocab_all.dat"
VOCAB_FILE_DICT="data/vocab_dict.dat"



class ZnQmDataset(Dataset):
    def __init__(self,DataDir,filelist,catfile,VOCAB):
        super(ZnQmDataset,self).__init__()
        self.allcat={}
        self.x_data=[]
        self.y_data=[]
        self.filelist=filelist
        self.vocab_dict={}

                
        with open(DATA_DIR+os.sep+TRAIN_LIST,"r") as f:
            allfiles=f.readlines()
        
            #for debug only
            # allfiles=allfiles[:100]
            for file in allfiles:
                file=file.strip()
                fullpath=DataDir+os.sep+os.sep+file 
                with open(fullpath,"r",encoding="utf-8") as f:
                    lines= f.readlines()  
                    lines=list(map(lambda x:x.replace("\n",""),lines))
                    lines=list(map(strip_chinese,lines))

                    newlines=[token   for token in lines if token != '' and token !=' ']
       
                    for item in newlines:
                        if item in self.vocab_dict:
                            self.vocab_dict[item]+=1
                        else:
                            self.vocab_dict[item]=1

                    nLines=len(newlines)
                    if  nLines <MAX_TOKEN :
                        newlines.extend([""]*(MAX_TOKEN-nLines))
                    else:
                        newlines=newlines[:MAX_TOKEN]

                                     
                    for item in newlines:
                        if item in self.vocab_dict:
                            self.vocab_dict[item]+=1
                        else:
                            self.vocab_dict[item]=1

                    nLines=len(newlines)
                    if  nLines <MAX_TOKEN :
                        newlines.extend([""]*(MAX_TOKEN-nLines))
                    else:
                        newlines=newlines[:MAX_TOKEN]

                    self.x_data.append(newlines)
                    self.y_data.append(self.allcat[str(Path(fullpath).suffix)[1:]])
        self.y_data=torch.tensor(self.y_data)

       


       
    
    def __len__(self):
        return len (self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    

def doTrain(ds,WORD_LIST):
    pass


if __name__=="__main__":
    
    VOCAB={}

   
    if(os.path.exists(DATASET_FILE)):
        ds=load_var(DATASET_FILE)
    else:
        ds=ZnQmDataset(DATA_DIR+os.sep+DST_DIR+os.sep+TRAIN_DIR,DATA_DIR+os.sep+TRAIN_LIST,DATA_DIR+os.sep+JSON_FILE,VOCAB)
        save_var(VOCAB,VOCAB_FILE)
        save_var(ds,DATASET_FILE)