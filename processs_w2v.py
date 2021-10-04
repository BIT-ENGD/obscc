
from  numba  import jit,njit
from numba.core.types.misc import Object, PyObject
from numba.experimental import jitclass  
import numba as nb #导入numba
from numba.typed import List #导入列表类型
import time 
import random 
import numpy as np  #用于类型转换
from numba import types, typed
import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from striphtml import strip_tags
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import os 
import re
# numba 0.54 
#
#
# struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)]) #转换数据结构
# >>> ty = numba.from_dtype(struct_dtype)
#
spec = [
    ("dir",types.unicode_type),
    ("ext",types.unicode_type),
    ("mylist",types.ListType(types.int64))
]

def ConvertStr(str):
    return np.frombuffer(bytes(str,"ascii"), dtype='uint8')

@jitclass(spec)
class  ZnJitTest(object):
    def __init__(self,dir,ext,myList):
        self.dir=dir
        self.ext=ext
        self.mylist=typed.List(myList)

    def TestFn(self):

        mylist=[random.randint(1,1000) for i in range(1000000000)]
        
        return mylist


@jitclass
class ZnUtility(object):
    def __init__(self):
        pass

    def removeRedundant(self,myList):
        myList=[  item.strip()    for item in myList  if item.strip() != ""]
        return myList



DATA_DIR="/data/ai/scc"
ORI_DIR="files"
DST_DIR="clean"
class ProcessWord2vec(object):
    def __init__(self,datadir) :
        self.newobj=ZnUtility()
        self.datadir=datadir
        g=os.walk(datadir)
        filenum=0
        contetns=[]
        for path,dirs,filelists in g:
            for file in filelists:
                fullfile=os.path.join(path,file)
                fileinfo=os.path.splitext(fullfile)
                ext=fileinfo[-1][1:]
                if(ext != "cpp"):
                    continue 

                with open(fullfile,"r") as f:
                    allines=f.readlines()
                    if  len(allines) ==0:
                        continue
                    allines=self.newobj.removeRedundant(allines)
                    contetns.append(allines)
                
                
                


                    
        
        print(filenum)

                
    def GetKeyWordSerial(self,strSrcCode):
        strPattern=r"([A-Za-z0-9_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`\"'])"
        pattern1=re.compile(strPattern,re.I)
        result=pattern1.findall(strSrcCode)
        return result
    
    def run(self):
        pass 


start=time.time()

ob=ProcessWord2vec(os.path.join(DATA_DIR,DST_DIR))
ml=ob.run()
end=time.time()
print(end-start)


