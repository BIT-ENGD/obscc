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
import threading 
import math 
import re
'''
zpobj=zp()

zpobj.finish()
zpobj.error()

'''


DATA_DIR="/data/linux/scc"
ORI_DIR="files"
DST_DIR="clean"
TRAIN_DIR="train"
VALID_DIR="valid"
TEST_DIR="test"

TRAIN_LIST="train.lst"
VALID_LIST="valid.lst"
TEST_LIST="test.lst"

CAT_FILE="languages.json"

THREAD_NUM=32



class ZnPreprocess(threading.Thread):
    def __init__(self,DataDir,DstDir,threadid,threadnum):
        super(ZnPreprocess,self).__init__()
        self.src=DataDir
        self.dst=DstDir

        self.threadid=threadid
        self.threadnum=threadnum

        if threadid ==0:
            if Path(self.dst).exists():
                shutil.rmtree(self.dst,ignore_errors=True)

            os.mkdir(self.dst)
            os.mkdir(self.dst+os.sep+TRAIN_DIR)
            os.mkdir(self.dst+os.sep+VALID_DIR)
            os.mkdir(self.dst+os.sep+TEST_DIR)

        

            
    def run(self):

        with open(DATA_DIR+os.sep+TRAIN_LIST,"r") as f:
             alltrains=f.readlines()
        alltrains=[file.strip() for file in alltrains]

        with open(DATA_DIR+os.sep+VALID_LIST,"r") as f:
             allvalids=f.readlines()
        allvalids=[file.strip() for file in allvalids]

        with open(DATA_DIR+os.sep+TEST_LIST,"r") as f:
             alltests=f.readlines()
        alltests=[file.strip() for file in alltests]

        dirs=[TRAIN_DIR,VALID_DIR,TEST_DIR]

        filelists=[alltrains,allvalids,alltests]

        for i in range(len(dirs)):
                
            self.filenumber=math.ceil(len(filelists[i])/self.threadnum)

            if(self.threadnum == self.threadid+1 ):
                self.filelist=filelists[i][self.threadid*self.filenumber:]
            else:    
                self.filelist=filelists[i][self.threadid*self.filenumber:(self.threadid+1)*self.filenumber]
             
            for file in self.filelist:
                oldfile=self.src+os.sep+dirs[i]+os.sep+file 
                newfile=self.dst+os.sep+dirs[i]+os.sep+file 
                self.ProcessSrcFile(oldfile,newfile)
                
    def GetKeyWordSerial(self,strSrcCode):
        strPattern=r"([A-Za-z0-9_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`\"'])"
        pattern1=re.compile(strPattern,re.I)
        result=pattern1.findall(strSrcCode)
        return result

    def ProcessSrcFile(self,src,dst):
        with open(src,encoding="utf-8") as f:
            srccontent=f.read()
            with open(dst,mode="w",encoding="utf-8") as nf:
                alllines=self.GetKeyWordSerial(srccontent)
                nf.write("\n".join(alllines))



class ZnQmDataset(Dataset):
    def __init__(self,DataDir,catfile,WORD_LIST):
        super(ZnQmDataset,self).__init__()
        self.allcat={}
        self.data_x=[]
        self.data_y={}
        with open(catfile,"r") as f:
            catlist=json.load(f)
            for id,item in enumerate(catlist.values()):
                self.allcat[item]=id
            
                
            with open(DATA_DIR+os.sep+TRAIN_LIST,"r") as f:
                allfiles=f.readlines()
            
            for file in allfiles:
                file=file.strip()
                fullpath=DataDir+os.sep+os.sep+file 
                #print(fullpath)

    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass





def doTrain(ds,vocab):
    pass

if __name__=="__main__":

    bPrepro=False  #　a switch for  the preprocessing task 

    if bPrepro:
        threads=[]
        for i in range(THREAD_NUM):
            thread=ZnPreprocess(DATA_DIR+os.sep+ORI_DIR,DATA_DIR+os.sep+DST_DIR,i,THREAD_NUM)
            thread.start()
            threads.append(thread)

        for subthread in threads:
            subthread.join()

    WORD_LIST=dict()

    ds=ZnQmDataset(DATA_DIR+os.sep+ORI_DIR+os.sep+TRAIN_DIR,DATA_DIR+os.sep+CAT_FILE,WORD_LIST)
    doTrain(ds,WORD_LIST)