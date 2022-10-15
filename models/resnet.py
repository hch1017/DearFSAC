import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm
import torch.utils.model_zoo as model_zoo

import math
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResnetHyper(nn.Module):
    def __init__(self, embedding_dim, args, hidden_dim=100, spec_norm=False, n_hidden=1):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(num_embeddings=self.args.num_users, embedding_dim=embedding_dim)
        
        layers = [spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim)]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 3 * 16 * 3 * 3)
        self.c1_bias = nn.Linear(hidden_dim, 3)
        
        # 很多要调的参数 plane inplane block.expansion kernel_size
        self.b1c1_weights = nn.Linear(hidden_dim, 16*16)
        self.b1c1_weights = nn.Linear(hidden_dim, 16)
        self.b1c2_weights = nn.Linear(hidden_dim, 16*16*3*3)
        self.b1c2_weights = nn.Linear(hidden_dim, 16)
        self.b1c3_weights = nn.Linear(hidden_dim, 16*16*4)
        self.b1c3_weights = nn.Linear(hidden_dim, 16*4)
        
        self.b2c1_weights = nn.Linear(hidden_dim, 64*16)
        self.b2c1_weights = nn.Linear(hidden_dim, 16)
        self.b2c2_weights = nn.Linear(hidden_dim, 16*16*3*3)
        self.b2c2_weights = nn.Linear(hidden_dim, 16)
        self.b2c3_weights = nn.Linear(hidden_dim, 16*16*4)
        self.b2c3_weights = nn.Linear(hidden_dim, 16*4)


        self.l1_weights = nn.Linear(hidden_dim, self.args.num_classes * 64 * 4)
        self.l1_bias = nn.Linear(hidden_dim, self.args.num_classes)  

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()    
        self.args = args
        self.inplanes = 16 #out
        print(self.args.bottleneck)
        if self.args.bottleneck == True:
            n = int((self.args.depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((self.args.depth - 2) / 6)
            block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, self.args.num_classes)

#         elif dataset == 'imagenet':
#             blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
#             layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
#             assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

#             self.inplanes = 64
#             self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)
#             self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
#             self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
#             self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
#             self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
#             self.avgpool = nn.AvgPool2d(7) 
#             self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1): # blocks=2
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion #16*4 for now
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

#         elif self.dataset == 'imagenet':
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)
#             x = self.maxpool(x)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc(x)
    
        return x