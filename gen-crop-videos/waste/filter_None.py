import h5py
import numpy as np
from tqdm import tqdm
from joblib import delayed
from joblib import Parallel

h5 = h5py.File('clipsListFile.hdf5','r')
filt_h5 = h5py.File('filter_clipsListFile.hdf5','w')
keys = h5.keys()

def write_wrapper(i):
    key = keys[i]
    if np.array(h5[key]['clip']).shape == (3,32,224,224):
	filt_h5[key+'/clip'] = np.array(h5[key]['clip'])
	filt_h5[key+'/label'] = np.array(h5[key]['label'])
    else:
    #print(np.array[h5[key]['clip']].shape)
	print(key)
	print (i*100.0/len(keys))

Parallel(n_jobs=20,backend="threading")(delayed(write_wrapper)(i) for i in range(len(keys)))
h5.close()
filt_h5.close()
