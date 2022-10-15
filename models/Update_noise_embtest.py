#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateTest(object):
    def __init__(self, args, dataset=None, idxs=None, flag=False, idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.flag = flag
        self.idx = idx #如果号数小于一个值，就给这个数据集加噪声
    def train(self, net):        
        net.train()
        # train and update
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        if self.flag:
#             print('flag')
            for iter in range(self.args.local_chosen_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
#                     if self.idx < 30:
                    images = images + self.args.noise_factor * torch.randn(*images.shape)
                    images = np.clip(images, 0., 1.)
                    labels = torch.tensor(random.sample(range(0,10),int(self.args.num_users*self.args.frac)))

                    w_glob = net.state_dict()
                    wmax = torch.max(w_glob['fc2.weight'])
                    wmin = torch.min(w_glob['fc2.weight'])
                    w_glob['fc2.weight'] += self.args.noise_factor * torch.randn(*w_glob['fc2.weight'].shape)
                    w_glob['fc2.weight'] = np.clip(w_glob['fc2.weight'], wmin, wmax)
                    bmax = torch.max(w_glob['fc2.bias'])
                    bmin = torch.min(w_glob['fc2.bias'])
                    w_glob['fc2.bias'] += self.args.noise_factor * torch.randn(*w_glob['fc2.bias'].shape)
                    w_glob['fc2.bias'] = np.clip(w_glob['fc2.bias'], wmin, wmax)
                    net.load_state_dict(w_glob)
                    
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                    break
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        else: 
            if self.args.local_ep > 0:
                for iter in range(self.args.local_ep):
                    batch_loss = []
                    for batch_idx, (images, labels) in enumerate(self.ldr_train):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
#                         if self.idx < self.args.noise_factor*self.args.num_users:
                        if self.idx < 50:
                  
                            w_glob = net.state_dict()
                            wmax = torch.max(w_glob['fc2.weight'])
                            wmin = torch.min(w_glob['fc2.weight'])
                            w_glob['fc2.weight'] += self.args.noise_factor * torch.randn(*w_glob['fc2.weight'].shape)
                            w_glob['fc2.weight'] = np.clip(w_glob['fc2.weight'], wmin, wmax)
                            bmax = torch.max(w_glob['fc2.bias'])
                            bmin = torch.min(w_glob['fc2.bias'])
                            w_glob['fc2.bias'] += self.args.noise_factor * torch.randn(*w_glob['fc2.bias'].shape)
                            w_glob['fc2.bias'] = np.clip(w_glob['fc2.bias'], wmin, wmax)
                            net.load_state_dict(w_glob)
                        net.zero_grad()
                        log_probs = net(images)
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        if self.args.verbose and batch_idx % 10 == 0:
                            print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                       100. * batch_idx / len(self.ldr_train), loss.item()))
                        batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                return net.state_dict(), 0
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

