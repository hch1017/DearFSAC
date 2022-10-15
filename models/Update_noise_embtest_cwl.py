#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdateNoise(object):
    def __init__(self, args, dataset=None, idxs=None, add_noise = False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.add_noise = add_noise

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.add_noise==True:
                    labels = torch.tensor(random.sample(range(0, 10), labels.shape[0])).to(self.args.device)
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
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(net)
    
    
#                         images = images + self.args.noise_factor * torch.randn(*images.shape)
#                         images = np.clip(images, 0., 1.)

#                         labels = torch.tensor(random.sample(range(0,10),int(self.args.num_users*self.args.frac)))

#                         w_glob = net.state_dict()
#                         wmax = torch.max(w_glob['fc2.weight'])
#                         wmin = torch.min(w_glob['fc2.weight'])
#                         w_glob['fc2.weight'] += self.args.noise_factor * torch.randn(*w_glob['fc2.weight'].shape)
#                         w_glob['fc2.weight'] = np.clip(w_glob['fc2.weight'], wmin, wmax)
#                         bmax = torch.max(w_glob['fc2.bias'])
#                         bmin = torch.min(w_glob['fc2.bias'])
#                         w_glob['fc2.bias'] += self.args.noise_factor * torch.randn(*w_glob['fc2.bias'].shape)
#                         w_glob['fc2.bias'] = np.clip(w_glob['fc2.bias'], wmin, wmax)
#                         net.load_state_dict(w_glob)

