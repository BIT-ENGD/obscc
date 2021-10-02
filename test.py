import glob
import pickle as pk
from numba import jit,njit
from numba.experimental import jitclass
import time

from torch.utils import data 

datafile="data/vocab_all.dat"

@jit
def LoadData(datafile):

    f=open(datafile,"rb")
    allcat=pk.load(f)
    print(len(allcat))
    f.close()
    return allcat

#@jit 
def enumdata(data):
    total=0
    for item in data:
        total+=data[item]
    print(total)

start=time.time()

enumdata(LoadData(datafile))
end=time.time()
print(end-start)