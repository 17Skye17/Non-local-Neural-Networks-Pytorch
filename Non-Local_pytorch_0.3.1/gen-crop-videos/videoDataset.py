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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RandomCrop(object):
    """Crop randomly the frames in a clip.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
			self.clip = clip

    def __call__(self):

        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top: top + new_h,
               left: left + new_w]

        return clip


class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, clipsListFile, channels, frames_num, xSize, ySize, mean, std, transform=None):
        """
		Args:
			clipsList (string): Path to the clipsList file with labels.
			rootDir (string): Directory with all the videoes.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
			mean: Mean valuse of the training set videos over each channel
		"""
        with open(clipsListFile, "rb") as fp:   # Unpickling
            clipsList = pickle.load(fp)

        self.clipsList = clipsList
        #self.rootDir = rootDir
        self.channels = channels
        self.frames_num = frames_num
        self.xSize = xSize
        self.ySize = ySize
		self.mean = mean
		self.std = std
		self.transform = transform

	def __len__(self):
		return len(self.clipsList)
    
	def randomCrop(self,frame):
    	h,w = frame.size()[2:]
		new_h,new_w = self.xSize,self.ySize
	
		top = np.random.randint(0 , h - new_h)
	
		left = np.random.randint(0 , w - new_w)
	
		clip = frame[:,:,top:top + new_h, left:left + new_w]
	
		normalized_clip = (clip - self.mean)/self.std
		return normalized_clip

    def readVideo(self, videoFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
		failedClip = False
		if width == 0 or height == 0 :
	    	clips = []
	    	failedClip = True
	    	print ("%s  size = 0"%videoFile)
	    	return clips,failedClip

		frames = torch.FloatTensor(self.channels, nFrames, height, width)
	
		if width < self.xSize or height < self.ySize:
	    	failedClip = True

		for f in range(nFrames):
	    	ret, frame = cap.read()
	    	if ret:
				frame = torch.from_numpy(frame)
			# to CHW
				frame = frame.permute(2,0,1)
				frames[:,f,:,:] = frame
	    	else:
				print("Skipped!")
				failedClip = True
				break
		# sample 64 clip then drop every other frame
		random_nums = (np.random.rand(self.frames_num)*nFrames).astype(np.int32)
		samples_frames = torch.FloatTensor(self.channels,self.frames_num,height,width)

		for i in range(self.frames_num):
	    	samples_frames[:,i,:,:] = frames[:,random_nums[i],:,:]
	
		final_frames = torch.FloatTensor(self.channels,32,height,width)

		for i in range(32):
	    	if i%2 == 0:
				final_frames[:,i,:,:] = samples_frames[:,i,:,:]
        
		if failedClip == False:
	    	clips = self.randomCrop(final_frames)
		else:
	    	clips = []
        return clips, failedClip

	def __getitem__(self, idx):
		videoFile = self.clipsList[idx][0]
		clip, failedClip = self.readVideo(videoFile)
	
		if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.clipsList[idx][1], 'failedClip': failedClip}

        return sample
