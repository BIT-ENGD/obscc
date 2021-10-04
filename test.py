import glob
import pickle as pk
#from numba import jit,njit
#from numba.experimental import jitclass
import time
import os
#from torch.utils import data 
import matplotlib.pylab as plt 
import matplotlib 

datafile="data/vocab_all.dat"
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

#@jit6
def LoadData(datafile):

    f=open(datafile,"rb")
    allcat=pk.load(f)
    print(len(allcat))
    f.close()
    return allcat

def save_var(varname,filename):
    with open(filename,"wb") as f:
        pk.dump(varname,f)

def load_var(filename):
    with open(filename,"rb") as f:
        return pk.load(f)


start=time.time()
vocab_all=load_var(os.path.join(DATA_DIR,VOCAB_FILE_REAL))
#vocab_dict=load_var(os.path.join(DATA_DIR,VOCAB_FILE_DICT))

print("len of vocab_all {}".format(len(vocab_all)))
maxfreq=max(vocab_all.values())
print("max freqencｅ",maxfreq)
#print("len of vocab_dict {}".format(len(vocab_dict)))
'''

nMax=0
def  mystatic(n):
    global nMax
    if( n>nMax):
        nMax=n
    


newlist=list(map(mystatic,vocab_all.values()))

print("max freqence 2:",nMax)

'''
# data=vocab_all.values()

# #data = [11,16,17,18,14,5,12,12,20]

# #matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False    

# plt.hist(data, bins=100, rwidth=0.5,facecolor="blue", edgecolor="black", alpha=0.7)
# plt.xlabel("区间")
# # 显示纵轴标签
# plt.ylabel("频数/频率")
# # 显示图标题
# plt.title("频数/频率分布直方图")
# plt.show()

allcat=load_var(CAT_FILE)
print(allcat)
end=time.time()
duration=end-start

print("duration:",duration)