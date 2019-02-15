import pickle
import torch
from torch.autograd import Variable
from lib.network import ResNet
import copy

model_path='/home/skye/DeepLearningPJ/video-nonlocal-net/checkpoints/c2d_baseline_32x2_IN_pretrain_400k.pkl'

model_weights = pickle.load(open('/home/skye/DeepLearningPJ/video-nonlocal-net/checkpoints/c2d_baseline_32x2_IN_pretrain_400k.pkl','rb'),encoding='latin1')
caffe_data = model_weights['blobs']
#keys = model_weights['blobs'].keys()
#print(len(keys))

def argparser(caffe_params,pytorch_params):
    assert len(caffe_params) == len(pytorch_params),'number of caffe params={}  vs pytorch params={}'.format(len(caffe_params),len(pytorch_params))
    name_map = {}
    new_map = {}
    name_map['layer1'] = 'res2'
    name_map['layer2'] = 'res3'
    name_map['layer3'] = 'res4'
    name_map['layer4'] = 'res5'
    name_map['downsample'] = 'branch1'
    name_map['running_mean'] = 'rm'
    name_map['running_var'] = 'riv'
    
    new_map['fc.weight'] = 'pred_w'
    new_map['fc.bias'] = 'pred_b'
    new_map['conv1.weight'] = 'conv1_w'
    new_map['bn1.weight'] = 'res_conv1_bn_s'
    new_map['bn1.bias'] = 'res_conv1_bn_b'
    new_map['bn1.running_mean'] = 'res_conv1_bn_rm'
    new_map['bn1.running_var'] = 'res_conv1_bn_riv'
    layers = ['layer1','layer2','layer3','layer4']
    
    for key in pytorch_params:
        for layer in layers:
            py_items = key.split('.')
            if layer in key:
                py_items = key.split('.')
                py_items[0] = name_map[py_items[0]]
                temp = copy.deepcopy(py_items)
                    
                if py_items[2] == 'downsample':
                    py_items[2] = 'branch1'
                    if py_items[3] == '0':
                        py_items[3] = 'w'
                        py_items.remove('weight')
                    if py_items[3] == '1':
                        py_items[3] = 'bn'
                        if py_items[4] == 'weight':
                            py_items[4] = 's'
                        if py_items[4] == 'bias':
                            py_items[4] ='b'
                        if py_items[4] == 'running_mean':
                            py_items[4] = 'rm'
                        if py_items[4] == 'running_var':
                            py_items[4] = 'riv'

                else:
                    if temp[2][-1] == '1':
                        py_items[2] = 'branch2a'
                    if temp[2][-1] == '2':
                        py_items[2] = 'branch2b'
                    if temp[2][-1] == '3':
                        py_items[2] = 'branch2c'
                    if temp[2][:-1] == 'conv':
                        py_items[3] = 'w'
                    elif temp[2][:-1] == 'bn':
                        py_items[3] = 'bn'
                        if temp[3] == 'weight':
                          py_items.append('s')
                        if temp[3] == 'bias':
                          py_items.append('b')
                 
                if temp[3] == 'running_mean':
                    py_items.append('rm')
                if temp[3] == 'running_var':
                    py_items.append('riv')


                new_param = py_items[0]
                for item in py_items[1:]:
                    new_param =  new_param + '_' + item
                new_map[key] = new_param
    return new_map

def check_param(name_map,caffe_params,pytorch_params):
    for key in name_map.keys():
        #if "downsample" in key:
            print (key)
            print (name_map[key])
            pytorch_shape = pytorch_params[key].shape
            caffe_shape = caffe_params[name_map[key]].shape
            assert pytorch_shape == caffe_shape, \
                    "pytorch param shape = {}  vs  caffe param shape = {}".format(pytorch_shape,caffe_shape)

caffe_params = {}
for key in caffe_data.keys():
        remove_voc = ['momentum','lr','iter']
        ADD_FLAG = True
        for voc in remove_voc:
            if voc in key:
                ADD_FLAG = False
        if ADD_FLAG:
            caffe_params[key] = caffe_data[key]
img = Variable(torch.randn(1,3,32,224,224))
net = ResNet.resnet50()

pytorch_params = {}

model_dict = net.state_dict()
for name in model_dict.keys():
#for name, param in net.named_parameters():
    if "num_batches_tracked" not in name:
        pytorch_params[name] = model_dict[name]

print (len(pytorch_params),len(caffe_params))
out = net(img)

name_map = argparser(caffe_params,pytorch_params)
check_param(name_map,caffe_params,pytorch_params)
