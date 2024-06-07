import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def norm_layer(channels, norm_type='gn'):
    if norm_type == 'bn':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'gn':
        return nn.GroupNorm(16, channels)
    elif norm_type == 'gn2':
        return nn.GroupNorm(2, channels)
    elif norm_type == 'gn4':
        return nn.GroupNorm(4, channels)
    elif norm_type == 'gn8':
        return nn.GroupNorm(8, channels)
    elif norm_type == 'gn32':
        return nn.GroupNorm(32, channels)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(channels)


class lenet(nn.Module):
    def __init__(self, norm_type=None, in_channels=1):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''



# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', norm_type='bn'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes, norm_type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     norm_layer(self.expansion * planes, norm_type=norm_type)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn', in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16, norm_type=norm_type)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, norm_type=norm_type)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_type=norm_type))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def num_rep(self):
        return 3

    def representation(self, x, ind=4, to_detach=False):
        bs = x.shape[0]
        res = []
        out = F.relu(self.bn1(self.conv1(x)))
        # res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        # if ind == 0:
        #     return res
        out = self.layer1(out)
        res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 1:
            return res
        out = self.layer2(out)
        res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 2:
            return res
        out = self.layer3(out)
        res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        if ind == 3:
            return res
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        res.append(out.detach().reshape([bs, -1]) if to_detach else out.reshape([bs, -1]))
        return res

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], norm_type=norm_type, **kwargs)


def resnet32(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], norm_type=norm_type, **kwargs)


def resnet44(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], norm_type=norm_type, **kwargs)


def resnet56(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], norm_type=norm_type, **kwargs)


def resnet110(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], norm_type=norm_type, **kwargs)


def resnet1202(norm_type='gn', **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], norm_type=norm_type, **kwargs)
