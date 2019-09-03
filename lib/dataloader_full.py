from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, dataloader
from pathlib import Path
import torch
from torch import optim, nn
import PIL 
import pydicom
import random

from functools import partial, reduce
from enum import Enum


# helper function from the competition
import mask_functions


def ls(p, re='*'):
    return [i for i in p.glob(re)]

def getImg(fpath):
    ds = pydicom.read_file( fpath) # read dicom image
    return ds.pixel_array, ds # get image array
    
def getMask(imgId, df, shape):
    """returns nd array mask"""
    df = df.loc[df['ImageId'] == imgId]
    rle_r = df['EncodedPixels'].tolist()
    
    rle = ''
    if len(rle_r) > 0:
        rle = rle_r[0].strip()

    mask = np.zeros([*shape])
    if not rle == '-1':
        mask = mask_functions.rle2mask(rle, *shape)
        mask = np.flip(mask, axis=0)
        mask = np.rot90(mask, -1)
    return mask

def getImageAndMask(fname, labels):
    imgf = fname 
    imgPath = str(imgf)
    imgId = imgf.stem
    img,_ = getImg( imgPath )
    mask = getMask(imgId, df=labels, shape=img.shape)
    return img, mask

def plot_multi(fn, size=[1,1], item_list=[]):
    fig, ax = plt.subplots(size[0],size[1], figsize=(40,60))
    
    ax = ax.flatten()
    for idx, a in enumerate(ax):
        if not idx > (len(item_list)-1):
            item = item_list[idx]
            fn(a, item)
                 
def plot_xray_mask(a, item, df, show_mask=True):
    img, mask = getImageAndMask(item, labels=df)
    a.imshow(img, cmap=plt.cm.bone)
    # if mask
    if not np.all(mask == 0): 
        a.set_title('Pneumothorax Present')
        a.imshow(mask, alpha=0.3, cmap='Reds')
        
def plot_xray(a, img):
    a.imshow(img, cmap=plt.cm.bone)

detach = lambda ten: ten.detach().numpy()[0]
    
def plot_xyhat(arr):
    item_list =[ detach(x) for x in arr ]

    plot_multi(plot_xray, size=[1,3], item_list=item_list)


def get_fnames(path):
    fnames = []
    for p in ls(path/'dicom-images-train'):
        for pp in ls(p):
            fnames.append(*ls(pp))
    return fnames

def get_labels(path):
    return pd.read_csv(path/'train-rle.csv').rename(columns={' EncodedPixels':'EncodedPixels'})



dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")



# lazy load the masks and images
class Pnuemo_ds(Dataset):
    def __init__(self, idxs, df):
        self.fnames = idxs
        self.labelsdf = df 
        self.len = len(self.fnames)
        
    def __getitem__(self, index):
        x,y = getImageAndMask(self.fnames[index], self.labelsdf)
        x = torch.Tensor(x).view(1, 1024, 1024).to(dev)
        y = torch.Tensor(y).view(1, 1024, 1024).to(dev)
        return x, y
        
    def __len__(self):
        return self.len
    
shrinkList = lambda l, s: l[0:(int(len(l)*s))]

def get_data(path, bs, split=0.8, shrink=1):
    fnames = get_fnames(path)
    df = get_labels(path)
    # split into val and training sets
    ln = len(fnames)
    trn_split = int(ln*split)
    # shrink data list
    random.seed(42)
    random.shuffle(fnames)

    trn_fnames_full = fnames[0:trn_split] 
    val_fnames_full = fnames[trn_split:]
    
    trn_fnames = shrinkList(trn_fnames_full, shrink) 
    val_fnames = shrinkList(val_fnames_full, shrink)  
    
    trn_ds = Pnuemo_ds(trn_fnames, df)
    trn_dl = dataloader.DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=0)

    val_ds = Pnuemo_ds(val_fnames, df)
    val_dl = dataloader.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    return {'dl': trn_dl, 'ds': trn_ds}, {'dl':val_dl,'ds':val_ds}



