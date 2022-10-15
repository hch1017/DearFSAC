#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
#     w_avg = list(range(len(w)))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for j in range(1, len(w)):
            w_avg[k] += w[j][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgMA(w):
    w_avg = list(range(len(w)))
    for i in range(len(w)):
        w_avg[i] = copy.deepcopy(w[i][0])
        for k in w_avg[i].keys():
            for j in range(1, len(w)):
                w_avg[i][k] += w[i][j][k]
            w_avg[i][k] = torch.div(w_avg[i][k], len(w[i]))
    #把w_avg内的元素对应求和再平均
    w_sum = w_avg[0]
    for i in range(1, len(w_avg)):
        for k in w_avg[i].keys():
            w_sum[k] += w_avg[i][k]
    for k in w_sum.keys():
        w_sum[k] /= len(w_avg)
    return w_sum

def FedPareto(w, action, choice):
    w_chosen = []
    for i in choice:
        w_chosen.append(w[i])
    w_avg = copy.deepcopy(w_chosen[0])
    for k in w_avg.keys():
        for i in range(0, len(w_chosen)):
            if i==0:
                w_avg[k] = action[i] * w_chosen[i][k]
            else:
                w_avg[k] += action[i] * w_chosen[i][k]
    return w_avg
