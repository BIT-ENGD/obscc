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


'''
zpobj=zp()

zpobj.finish()
zpobj.error()

'''


DATA_DIR="/data/linux/scc/files"
TRAIN_DIR="train"
VALID_DIR="valid"
