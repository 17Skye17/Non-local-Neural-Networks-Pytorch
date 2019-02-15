import torch
import torch.utils.data as Data
import torchvision
from lib.network import ResNet
#from lib.network import ResNet
from torch.autograd import Variable
from torch import nn
from videoDataset import videoDataset
import pickle
import time
import h5py
from checkpoints.check import argparser

def load_params(pretrained_model_path, model_params):
    caffe_weights = pickle.load(open(pretrained_model_path,'rb'),encoding='latin1')
    caffe_data = caffe_weights['blobs']
    caffe_params = {}
    remove_voc = ['momentum','lr','iter']
    for key in caffe_data.keys():
        ADD_FLAG = True
        for voc in remove_voc:
            if voc in key:
                ADD_FLAG = False
        if ADD_FLAG:
            caffe_params[key] = caffe_data[key]

    name_map = argparser(caffe_params, model_params) 
    
    state_dict = {}
    for key in model_params.keys():
        state_dict['module.'+key] = torch.FloatTensor(caffe_params[name_map[key]])
    return state_dict

def calc_acc(x, y):
    x = torch.max(x, dim=-1)[1]
    accuracy = (sum(x == y))*100.0 / x.size(0)
    return accuracy

pretrained_model_path = '/home/skye/DeepLearningPJ/video-nonlocal-net/checkpoints/c2d_baseline_32x2_IN_pretrain_400k.pkl'

hdf5_file = 'gen-crop-videos/filter_clipsListFile.hdf5'
video = videoDataset(hdf5_file,3,32,224,224)

train_loader = Data.DataLoader(dataset=video, batch_size=128, shuffle=True)
test_loader = Data.DataLoader(dataset=video, batch_size=128, shuffle=False)
#
train_batch_num = len(train_loader)
test_batch_num = len(test_loader)

net = ResNet.resnet50()
model_dict = net.state_dict()
model_params = {}
for name in model_dict.keys():
    if "num_batches_tracked" not in name:
        model_params[name] = model_dict[name]

print ("start to load pretrained model ....")
net_params = load_params(pretrained_model_path,model_params)
if torch.cuda.is_available():
    net = nn.DataParallel(net)
    net.cuda()


model_dict = net.state_dict()
net_params = {k:v for k,v in net_params.items() if k in model_dict.keys()}

model_dict.update(net_params)
net.load_state_dict(model_dict)

print ("parameter loaded!")
loss_func = nn.CrossEntropyLoss()

for epoch_index in range(1):
    st = time.time()
    print ("start to test...")
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
      for test_batch_index, sample_batch in enumerate(test_loader):
          img_batch = Variable(sample_batch['clip'])
          label_batch = Variable(sample_batch['label'])

          if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              label_batch = label_batch.cuda()

          predict = net(img_batch)
          acc = calc_acc(predict.cpu().data, label_batch.cpu().data)
          loss = loss_func(predict, label_batch)

          total_loss += loss
          total_acc += acc
          print ("loss = %.3f   acc = %.3f"%(loss,acc))

    mean_acc = total_acc / test_batch_num
    mean_loss = total_loss / test_batch_num

    print('[Test] epoch[%d/%d] acc:%.4f loss:%.4f\n'
          % (epoch_index, 100, mean_acc, mean_loss.data[0]))
