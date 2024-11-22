#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class SequentialWithInternalStatePrediction(nn.Sequential):
    """
    Adapted version of Sequential that implements the function predict_internal_states
    """

    def predict_internal_states(self, x):
        """
        applies the submodules on the input. Compared to forward, this function also returns
        all intermediate outputs
        """
        result = []
        for module in self:
            x = module(x)
            # We can define our layer as we want. We selected Convolutional and
            # Linear Modules as layers here.
            # Differs for every model architecture.
            # Can be defined by the defender.
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                result.append(x)
        return result, x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                #random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(torch.cuda.FloatTensor)
                # negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None):
        super(ResNet, self).__init__(name, created_time)
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)
        return SequentialWithInternalStatePrediction(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # for SDTdata
        # return F.softmax(out, dim=1)
        # for regular output
        return out

    def predict_internal_states(self, x):
        result = []
        x = self.conv1(x)
        result.append(x)
        x = F.relu(self.bn1(x))
        res, x = self.layer1.predict_internal_states(x)
        result += res
        res, x = self.layer2.predict_internal_states(x)
        result += res
        res, x = self.layer3.predict_internal_states(x)
        result += res
        res, x = self.layer4.predict_internal_states(x)
        result += res
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        result.append(x)

        return result

# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18(name=None, created_time=None):
    return ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(name), created_time=created_time)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

def get_model(data):
    if data == 'fmnist' or data == 'fedemnist':
        return CNN_MNIST()
    elif data == 'cifar10':
        return CNN_CIFAR()
               

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x   



class MnistNet(SimpleNet):
    # def __init__(self, name=None, created_time=None):
    #     super(MnistNet, self).__init__(f'{name}_Simple', created_time)

    #     self.conv1 = nn.Conv2d(1, 20, 5, 1)
    #     self.conv2 = nn.Conv2d(20, 50, 5, 1)
    #     self.fc1 = nn.Linear(4 * 4 * 50, 500)
    #     self.fc2 = nn.Linear(500, 10)
    #     # self.fc2 = nn.Linear(28*28, 10)

    def __init__(self):
        super(MnistNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        # x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # in_features = 28 * 28
        # x = x.view(-1, in_features)
        # x = self.fc2(x)

        # normal return:
        return F.log_softmax(x, dim=1)
        # soft max is used for generate SDT data
        # return F.softmax(x, dim=1)    

    def predict_internal_states(self, x):
        result = []
        x = self.conv1(x)
        result.append(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        result.append(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        result.append(x)
        x = F.relu(x)
        x = self.fc2(x)
        result.append(x)

        return result

# class MnistNet(nn.Module):
#     # def __init__(self, name=None, created_time=None):
#     #     super(MnistNet, self).__init__(f'{name}_Simple', created_time)

#     #     self.conv1 = nn.Conv2d(1, 20, 5, 1)
#     #     self.conv2 = nn.Conv2d(20, 50, 5, 1)
#     #     self.fc1 = nn.Linear(4 * 4 * 50, 500)
#     #     self.fc2 = nn.Linear(500, 10)
#     #     # self.fc2 = nn.Linear(28*28, 10)

#     def __init__(self):
#         super(MnistNet, self).__init__()

#         # self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         # self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         # self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         # self.fc2 = nn.Linear(500, 10)

#         self.features = SequentialWithInternalStatePrediction(
#             nn.Conv2d(1, 20, 5, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(20, 50, 5, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.classifier = SequentialWithInternalStatePrediction(
#             nn.Linear(4 * 4 * 50, 500),
#             nn.ReLU(inplace=True),
#             nn.Linear(500, 10),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(-1, 4 * 4 * 50)
#         x = self.classifier(x)
#         # x = F.relu(self.conv1(x))
#         # x = F.max_pool2d(x, 2, 2)
#         # x = F.relu(self.conv2(x))
#         # x = F.max_pool2d(x, 2, 2)
#         # x = x.view(-1, 4 * 4 * 50)
#         # # x = x.view(-1, 32 * 7 * 7)
#         # x = F.relu(self.fc1(x))
#         # x = self.fc2(x)

#         # in_features = 28 * 28
#         # x = x.view(-1, in_features)
#         # x = self.fc2(x)

#         # normal return:
#         return F.log_softmax(x, dim=1)
#         # soft max is used for generate SDT data
#         # return F.softmax(x, dim=1)
    
#     def predict_internal_states(self, x):
#         result, x = self.features.predict_internal_states(x)
#         x = x.view(-1, 4 * 4 * 50)
#         result += self.classifier.predict_internal_states(x)[0]
#         return result

class LoanNet(SimpleNet):
    def __init__(self, in_dim=91, n_hidden_1=46, n_hidden_2=23, out_dim=9, name=None, created_time=None):
    # def __init__(self, in_dim=97, n_hidden_1=46, n_hidden_2=23, out_dim=9, name=None, created_time=None):
        super(LoanNet, self).__init__(f'{name}_Simple', created_time)
        self.layer1 = SequentialWithInternalStatePrediction(nn.Linear(in_dim, n_hidden_1),
                                    nn.Dropout(0.5), # drop 50% of the neuron to avoid over-fitting
                                    nn.ReLU())
        self.layer2 = SequentialWithInternalStatePrediction(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Dropout(0.5),
                                    nn.ReLU())
        self.layer3 = SequentialWithInternalStatePrediction(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            print(x)
            raise ValueError()
        return x
    
    def predict_internal_states(self, x):
        result = []

        res, x = self.layer1.predict_internal_states(x)
        result += res
        res, x = self.layer2.predict_internal_states(x)
        result += res
        res, x = self.layer3.predict_internal_states(x)
        result += res

        return result