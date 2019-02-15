"""
PyTorch Video Dataset Class for loading videos using PyTorch
Dataloader. This Dataset assumes that video files are Preprocessed
 by being trimmed over time and resizing the frames.

Mohsen Fayyaz __ Sensifai Vision Group
http://www.Sensifai.com

If you find this code useful, please star the repository.
"""

from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pickle
import random
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, clipsListFile, channels, frames_num, xSize, ySize, transform=None):
       
        clipsList = h5py.File(clipsListFile,'r')
        self.keys = list(clipsList.keys())
        random.shuffle(self.keys)
        self.clipsList = clipsList
        #self.rootDir = roo
        self.channels = channels
        self.frames_num = frames_num
        self.xSize = xSize
        self.ySize = ySize
        self.transform = transform
	
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        video_clip = np.array(self.clipsList[str(key)]['clip'],dtype=np.float32)
        video_label = np.array(self.clipsList[str(key)]['label']).astype(float)
        if self.transform:
            clip = self.transform(clip)
	    
        sample = {'clip':torch.from_numpy(video_clip),'label':torch.tensor(video_label,dtype=torch.long)}
	
        return sample
