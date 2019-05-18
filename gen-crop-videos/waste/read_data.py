from videoDataset import RandomCrop
from videoDataset import videoDataset
import torch.utils.data as Data
import h5py
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

MEAN = 114.75
STD = 57.37
h5 = h5py.File('clipsListFile.hdf5','w')
video = videoDataset('clipsListFile.pkl',3,64,224,224,MEAN,STD)

print (type(video))
#Pickle.dump(video,open('cropped_clipsListFile.pkl','wb'),2)
#video = Pickle.load()
def write_wrapper(i):
    sample = video[i]
    clip = sample['clip']
    h5[str(i)+'/clip'] = clip
    
    label = sample['label']
    h5[str(i)+'/label'] = label
    print (i*100.0/len(video))

Parallel(n_jobs=20,backend="threading")(delayed(write_wrapper)(i) for i in range(len(video)))
#pbar = tqdm(total=len(video))
#for i in range(len(video)):
#    pbar.update(1)
#    sample = video[i]
h5.close()
#pbar.close()

#test_loader=Data.DataLoader(dataset=video,batch_size=1,shuffle=False)
#for sample in enumerate(test_loader):
#    print sample
