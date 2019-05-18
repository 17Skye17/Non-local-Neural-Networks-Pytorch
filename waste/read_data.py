from videoDataset import videoDataset
import torch.utils.data as Data
import h5py
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    batch = list(filter(lambda x:x is not None,batch))
    batch = list(filter(lambda x:x[0] is not None, batch))
    return default_collate(batch)

hdf5_file='sample-clipsListFile.hdf5'

video = videoDataset(hdf5_file,3,32,224,224)

dataloader = Data.DataLoader(dataset=video,batch_size=1,collate_fn=my_collate_fn,shuffle=False)
for sample in enumerate(dataloader):
    print sample

#for i in range(len(video)):
#    print np.array(video[i]['clip']).shape
#test_loader=Data.DataLoader(dataset=video,batch_size=2,shuffle=False)
#for sample in enumerate(test_loader):
    
#for sample in enumerate(test_loader):
#    print sample
