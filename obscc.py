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
import time
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

JSON_FILE="languages.json"

MODEL_NAME="src_cat.pth"
ONNX_MODEL_PATH="src_cat.onnx"

DATASET_FILE="ds.dat"
VOCAB_FILE="vocab.dat"
CAT_FILE="allcat.dat"

DATAFILE_PREFIX="ds_{}_{}.dat"
VOCABFILE_PREFIX="vocab_{}.dat"

THREAD_NUM=32

MAX_TOKEN =1000

MIN_WORD_FREQUENCE=3

LASTPART=-1  # -1 normal, 100000 test only

def save_var(varname,filename):
    with open(filename,"wb") as f:
        pk.dump(varname,f)

def load_var(filename):
    with open(filename,"rb") as f:
        return pk.load(f)



class ZnPreprocess(threading.Thread):
    def __init__(self,DataDir,DstDir,threadid):
        super(ZnPreprocess,self).__init__()
        self.src=DataDir
        self.dst=DstDir

        self.threadid=threadid
      
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
                
            self.filenumber=math.ceil(len(filelists[i])/THREAD_NUM)

            if(THREAD_NUM == self.threadid+1 ):
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



def strip_chinese(strs):
    if strs.find("STRSTUFF") > -1 and len(strs)>8:
        print(strs)
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return ""
    return strs

# get the list of unique vocab
class ZnQmProcessData(threading.Thread):
    def __init__(self,DataDir,catfile,filelist,threadid,vocab,bGetVocab):
        super(ZnQmProcessData,self).__init__()
        
        with open(DataDir+os.sep+".."+os.sep+filelist,"r") as f:
            self.allfiles=f.readlines()

        self.allfiles=self.allfiles[:LASTPART] # debug only
        self.datadir=DataDir+os.sep+TRAIN_DIR
        self.catfile=DataDir+os.sep+".."+os.sep+catfile
        self.threadid=threadid  
        self.vocab=vocab
        self.vocab_dict={}
        self.RungetVocab=bGetVocab
    
    def run(self):

        if self.RungetVocab:
            self.getvocab()
        else:
            self.gettraininfo()

    def getvocab(self):
        allcat={}
        if self.threadid == 0:
            with open(self.catfile,"r") as f:
                catlist=json.load(f)
                for id,item in enumerate(catlist.values()):
                    allcat[item]=id
            save_var(allcat,CAT_FILE)

        self.filenumber=math.ceil(len(self.allfiles)/THREAD_NUM)

        if(THREAD_NUM == self.threadid+1 ):
            filelist=self.allfiles[self.threadid*self.filenumber:]
        else:    
            filelist=self.allfiles[self.threadid*self.filenumber:(self.threadid+1)*self.filenumber]

        for file in filelist:
            file=file.strip()
            fullpath=self.datadir+os.sep+file 
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
        self.new_vocab=set()
        #self.new_vocab[""]=0
        for item in self.vocab_dict:
            if self.vocab_dict[item] >MIN_WORD_FREQUENCE:
                self.new_vocab.update(item)
              
        
        filename=VOCABFILE_PREFIX.format(self.threadid)
        save_var(self.new_vocab,filename)

        #VOCAB.update(new_vocab.keys()) # 赋值
        #VOCAB.add("") #添加空字符
  
    def gettraininfo(self):

        self.filenumber=math.ceil(len(self.allfiles[i])/THREAD_NUM)

        if(THREAD_NUM == self.threadid+1 ):
            self.filelist=self.allfiles[i][self.threadid*self.filenumber:]
        else:    
            self.filelist=self.allfiles[i][self.threadid*self.filenumber:(self.threadid+1)*self.filenumber]
        lines=[]
        vocab_dict={}
        for file in self.allfiles:
            file=file.strip()
            fullpath=self.datadir+os.sep+file 
            with open(fullpath,"r",encoding="utf-8") as f:
                lines= f.readlines()  
                lines=list(map(lambda x:x.replace("\n",""),lines))
                lines=list(map(strip_chinese,lines))

                newlines=[token   for token in lines if token != '' and token !=' ']

                for item in newlines:  # count the frequence of words
                    if item in vocab_dict:
                        vocab_dict[item]+=1
                    else:
                        vocab_dict[item]=1

                nLines=len(newlines)
                if  nLines <MAX_TOKEN :
                    newlines.extend([""]*(MAX_TOKEN-nLines))
                else:
                    newlines=newlines[:MAX_TOKEN]

                theline=[self.vocab[item] if item in self.vocab else 0  for item in newlines]
                lines.append(theline)
        length=len(lines)

        filename=DATAFILE_PREFIX.format(self.threadid,length)
        save_var(lines,filename)

        
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

    bPrepro=False  #　a switch for  the preprocessing task 

    if bPrepro:
        threads=[]
        for i in range(THREAD_NUM):
            thread=ZnPreprocess(DATA_DIR+os.sep+ORI_DIR,DATA_DIR+os.sep+DST_DIR,i,THREAD_NUM)
            thread.start()
            threads.append(thread)

        for subthread in threads:
            subthread.join()

#  VOCAB=set()

#   if os.path.exists(VOCAB_FILE):
#        VOCAB=load_var(VOCAB_FILE)



    
    bGenData=True
    if(bGenData):
        vocab={}
        runner=[]
        for i in range(THREAD_NUM):
            dp=ZnQmProcessData(DATA_DIR+os.sep+DST_DIR,JSON_FILE,TRAIN_LIST,i,vocab,True)
            dp.start()
            runner.append(dp)

        for task in runner:
            task.join()



    VOCAB={}

    if(os.path.exists(DATASET_FILE)):
        ds=load_var(DATASET_FILE)
    else:
        ds=ZnQmDataset(DATA_DIR+os.sep+DST_DIR+os.sep+TRAIN_DIR,DATA_DIR+os.sep+TRAIN_LIST,DATA_DIR+os.sep+JSON_FILE,VOCAB)
        save_var(VOCAB,VOCAB_FILE)
        save_var(ds,DATASET_FILE)

  

    WORDLIST={key:i for i,key in enumerate(VOCAB)}

  
    zpobj=zp()
    doTrain(ds,WORDLIST)

    zpobj.finish()
    print("done!!!")