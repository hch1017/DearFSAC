#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.utils import spectral_norm

#channel = args.num_channels  kernel = 10
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNNHyperMnist(nn.Module):
    def __init__(self, embedding_dim, args, hidden_dim=100, spec_norm=False, n_hidden=1):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(num_embeddings=self.args.num_users, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 10 * self.args.num_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 10)
        self.c2_weights = nn.Linear(hidden_dim, 20 * 10 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 20)
        self.l1_weights = nn.Linear(hidden_dim, 50 * 20 * 4 * 4)
        self.l1_bias = nn.Linear(hidden_dim, 50)
        self.l2_weights = nn.Linear(hidden_dim, self.args.num_classes * 50)
        self.l2_bias = nn.Linear(hidden_dim, self.args.num_classes)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(10, args.num_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(20, 10, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(50, 20 * 4 * 4),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(self.args.num_classes, 50),
            "fc2.bias": self.l2_bias(features).view(-1),
        })
        return weights


class CNNHyperCifar(nn.Module):
    def __init__(self, embedding_dim, args, hidden_dim=100, spec_norm=False, n_hidden=1):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(num_embeddings=self.args.num_users, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 6 * 3 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 6)
        self.c2_weights = nn.Linear(hidden_dim, 16 * 6 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 16)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 16 * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.args.num_classes * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.args.num_classes)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(6, 3, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(16, 6, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 16 * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.args.num_classes, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights

    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
    
    
class CNNCifarEmb(nn.Module):
    def __init__(self, input_dim, emb_feature_dim=100, dim_hidden=500):
        super(CNNCifarEmb, self).__init__()
        self.layer_input = nn.Linear(input_dim, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, emb_feature_dim)
    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x

    
class CNNMnistEmbcwl(nn.Module):
    def __init__(self, input_dim, args, emb_feature_dim=100, dim_hidden=512, hidden_dim=128,):
        super(CNNMnistEmbcwl, self).__init__()
        self.layer_input = nn.Linear(input_dim, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, emb_feature_dim)
        self.args = args
        self.emb_feature_dim = emb_feature_dim
        layers = [
            nn.Linear(self.emb_feature_dim, hidden_dim),
        ]
        layers.append(nn.ReLU(inplace=True))
#         layers.append(
#             nn.Linear(hidden_dim,hidden_dim),
#         )
        self.mlp = nn.Sequential(*layers)
        self.c1_weights = nn.Linear(hidden_dim, 10 * args.num_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 10)
        self.c2_weights = nn.Linear(hidden_dim, 20 * 10 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 20)
        self.l1_weights = nn.Linear(hidden_dim, 50 * 20 * 4 * 4)
        self.l1_bias = nn.Linear(hidden_dim, 50)
        self.l2_weights = nn.Linear(hidden_dim, self.args.num_classes * 50)
        self.l2_bias = nn.Linear(hidden_dim, self.args.num_classes)
        self.whether_noise1 = nn.Linear(self.emb_feature_dim, 32)
        self.whether_noise2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x

    def forward2(self, x):
        x = self.whether_noise1(x)
        x = self.whether_noise2(x)
        x = F.sigmoid(x)
        return x

#     def Reverse(self, x):
#         features = self.mlp(x)

#         weights = OrderedDict({
#             "conv1.weight": self.c1_weights(features).view(10, self.args.num_channels, 5, 5),
#             "conv1.bias": self.c1_bias(features).view(-1),
#             "conv2.weight": self.c2_weights(features).view(20, 10, 5, 5),
#             "conv2.bias": self.c2_bias(features).view(-1),
#             "fc1.weight": self.l1_weights(features).view(50, 20 * 4 * 4),
#             "fc1.bias": self.l1_bias(features).view(-1),
#             "fc2.weight": self.l2_weights(features).view(self.args.num_classes, 50),
#             "fc2.bias": self.l2_bias(features).view(-1),
#         })
#         return weights

    def Reverse_raw_emb(self, x):
        features = self.mlp(x)
        output_features = torch.cat([
            self.c1_weights(features),
            self.c1_bias(features),
            self.c2_weights(features),
            self.c2_bias(features),
            self.l1_weights(features),
            self.l1_bias(features),
            self.l2_weights(features),
            self.l2_bias(features),
            # self.whether_noise(features)
        ], 1)
#         whether_noise = self.whether_noise1(output_features)
#         whether_noise = self.whether_noise2(whether_noise)
#         whether_noise = F.sigmoid(whether_noise)
        return output_features

    
class CNNMnistEmb(nn.Module):
    def __init__(self, input_dim, args, emb_feature_dim=100, dim_hidden=500, hidden_dim=100,):
        super(CNNMnistEmb, self).__init__()
        self.layer_input = nn.Linear(input_dim, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, emb_feature_dim)
        self.args = args
        layers = [
            nn.Linear(emb_feature_dim, hidden_dim),
        ]
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Linear(hidden_dim,hidden_dim),
        )
        self.mlp = nn.Sequential(*layers)
        self.c1_weights = nn.Linear(hidden_dim, 10 * args.num_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 10)
        self.c2_weights = nn.Linear(hidden_dim, 20 * 10 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 20)
        self.l1_weights = nn.Linear(hidden_dim, 50 * 20 * 4 * 4)
        self.l1_bias = nn.Linear(hidden_dim, 50)
        self.l2_weights = nn.Linear(hidden_dim, self.args.num_classes * 50)
        self.l2_bias = nn.Linear(hidden_dim, self.args.num_classes)
        
    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x

    def Reverse(self, x):
        features = self.mlp(x)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(10, self.args.num_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(20, 10, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(50, 20 * 4 * 4),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(self.args.num_classes, 50),
            "fc2.bias": self.l2_bias(features).view(-1),
        })
        return weights

    def Reverse_raw_emb(self, x):
        features = self.mlp(x)
        output_features = torch.cat([
            self.c1_weights(features),
            self.c1_bias(features),
            self.c2_weights(features),
            self.c2_bias(features),
            self.l1_weights(features),
            self.l1_bias(features),
            self.l2_weights(features),
            self.l2_bias(features)
        ], 1)

        return output_features
    
class CNNCifarEmbReverse(nn.Module):
    def __init__(self, args, input_dim=100, hidden_dim=100, spec_norm=False, n_hidden=1):
        super(CNNCifarEmbReverse, self).__init__()
        self.args = args
        layers = [
            spectral_norm(nn.Linear(input_dim, hidden_dim)) if spec_norm else nn.Linear(input_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 6 * 3 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 6)
        self.c2_weights = nn.Linear(hidden_dim, 16 * 6 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 16)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 16 * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.args.num_classes * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.args.num_classes)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, x):
        features = self.mlp(x)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(6, 3, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(16, 6, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 16 * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.args.num_classes, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights    
    
    
    
    #channel=3 kernel=6 
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         return x
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x   
    
class CNNUsps(nn.Module):
    def __init__(self, args):
        super(CNNUsps, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


class CNNFashionmnist(nn.Module): # extend nn.Module class of nn
    def __init__(self, args):
        super(CNNFashionmnist, self).__init__() # super class constructor
        self.conv1 = nn.Conv2d(args.num_channels, out_channels=6, kernel_size=(5,5))
        self.batchN1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5,5))
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.batchN2 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=args.num_classes)
        
        
        
    def forward(self, t): # implements the forward method (flow of tensors)
        
        # hidden conv layer 
        t = self.conv1(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)
        t = self.batchN1(t)
        
        # hidden conv layer
        t = self.conv2(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)
        
        # flatten
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.batchN2(t)
        t = self.fc2(t)
        t = F.relu(t)
        
        # output
        t = self.out(t)
        
        return t  
    