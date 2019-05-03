from torchvision.models import resnet
from torch import nn

class Fake(nn.Module):
    def __init__(self, x):
        super().__init__()

    def forward(self, x):
        return x

class CorrNet(resnet.ResNet):
    '''post-process correlation feature maps
    '''
    def __init__(self, inplanes, planes = 256):
        '''
        inplanes : number of channels of input tensor
        planes : number of channels in this network
        '''
        super(resnet.ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=1)
        self.bn1 = self._norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        block = resnet.BasicBlock
        
        self.inplanes = planes
        self.layer1 = self._make_layer(block, planes, 2)
        self.layer2 = self._make_layer(block, planes * 2, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.avgpool(out)
        return out.reshape((out.size(0), -1))
