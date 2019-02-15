import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=(1,1,1), padding=(0,1,1), downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Conv3d(inplanes, planes, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)
        self.bn1=nn.BatchNorm3d(planes)
        self.conv2=nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=stride, padding=padding, bias=False)
        self.bn2=nn.BatchNorm3d(planes)
        self.conv3=nn.Conv3d(planes, planes*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)
        self.bn3=nn.BatchNorm3d(planes*4)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual=self.downsample(x)

        out +=residual
        out=self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,block,layers, num_classes=400, zero_init_residual=False):
        self.inplanes = 64
        super(ResNet,self).__init__()
        self.conv1=nn.Conv3d(3,64,kernel_size=(1,7,7),stride=(2,2,2), padding=(0,3,3), bias=False)
        self.bn1=nn.BatchNorm3d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool1=nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(0,0,0))
        self.layer1=self._make_layer(block, 64, layers[0], downsample_padding=(0,0,0))
        self.maxpool2=nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(0,0,0))
        self.layer2=self._make_layer(block, 128, layers[1], stride=(2,2,2), padding=(2,1,1))
        self.layer3=self._make_layer(block, 256, layers[2], stride=(2,2,2), padding=(2,1,1))
        self.layer4=self._make_layer(block, 512, layers[3], stride=(2,2,2), padding=(2,1,1))
        self.avgpool=nn.AvgPool3d((4,7,7), stride=(1,1,1))
        self.fc=nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=(1,1,1), padding=(0,1,1), downsample_padding=(2,0,0)):
        downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes*block.expansion, kernel_size=(1,1,1), stride=stride,padding=downsample_padding, bias=False), nn.BatchNorm3d(planes* block.expansion))

        layers =[]
        layers.append(block(self.inplanes, planes, stride, padding, downsample))
        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool1(x)

        x=self.layer1(x)
        x=self.maxpool2(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)

        x=x.view(x.size(0), -1)
        x=self.fc(x)

        return x
    

    def resnet50(**kwargs):
        model = ResNet(Bottleneck,[3,4,6,3], **kwargs)
        return model


if __name__=='__main__':
    import torch
    from torch.autograd import Variable
    img = Variable(torch.randn(1,3,32,224,224))
    net = ResNet.resnet50()
    count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count += 1
            print(name)
    print (count)
    out = net(img)
    print(out.size())

