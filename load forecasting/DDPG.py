import pickle
from itertools import count

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import grad
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.utils import spectral_norm

import argparse


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for j in range(1, len(w)):
            w_avg[k] += w[j][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000, help="rounds of training")

    #嵌入向量的训练轮次
    parser.add_argument('--emb_train_epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--emb', default=True)
    
    #验证RL和Fedavg哪个更好的验证轮次
    parser.add_argument('--validation_epochs', type=int, default=50, help="rounds of training")
    
    #有多少个local client
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    
    #每次选多少个local client参与训练
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    
    #RL的学习率和衰减率
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate (default: 0.01)")
    parser.add_argument('--lr_decay', type=float, default=1, help="lr decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    
    #输出的分类个数
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
      
    #resnet parameter

    args = parser.parse_args(args=[])
    return args

args = args_parser()
args.device = 'cuda:1'

class MemoryBuffer: # MemoryBuffer类实现的功能：buffer内采样，往buffer里塞（sars）

    def __init__(self, size):
        self.buffer = deque(maxlen=size) #buffer设置为双端队列
        self.maxSize = size
        self.len = 0
        
    def state_reco(self, s):
        s_1 = [i[0] for i in s]
        s_2 = [i[1] for i in s]
        s_3 = [i[2] for i in s]
        return [torch.cat(s_1),torch.cat(s_2),torch.cat(s_3)]

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count) # 随机取样

        s_arr = [arr[0] for arr in batch]
        a_arr = torch.cat([arr[1] for arr in batch])
        r_arr = torch.tensor([arr[2] for arr in batch]).reshape(-1,1)
        s1_arr = [arr[3] for arr in batch]

        return self.state_reco(s_arr), a_arr, r_arr, self.state_reco(s1_arr)

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s,a,r,s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
        
class Actor(nn.Module):
    def __init__(self, parameter_dim, loss_dim, action_dim):
        super(Actor, self).__init__()
        self.parameter_dim = parameter_dim
        self.loss_dim = loss_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(parameter_dim, action_dim)
        self.fc2 = nn.Linear(action_dim*(int(args.num_users*args.frac+2)), self.action_dim)
#         self.ln2 = nn.LayerNorm(256)
#         self.fc3 = nn.Linear(256, self.action_dim)
#         self.fc4 = nn.Linear(256, 1)
        
    def forward(self, parameters, last_loss, last_weight):
        parameter_lst = []
        for i in range(self.action_dim):
            parameter_lst.append(self.fc1(parameters[:,i,:]))
        parameter_layer = torch.cat(parameter_lst,dim=1)
        x = torch.cat([parameter_layer,last_loss, last_weight],dim=1)
#         x = self.fc2(x1.clone()) # 256
#         x = self.ln2(x)
#         x3 = F.sigmoid(self.fc3(x2))
#         acc = F.sigmoid(self.fc4(x))
#         output = F.softmax(self.fc2(x),dim=1)
#         x2 = F.sigmoid(x2.clone())
#         for i in range(len(x2)):
#             tmp = torch.sum(x2[i])
#             x2[i] = x2[i]/tmp
        x = self.fc2(x)
        action = F.softmax(x,dim=1)
        return action

class Critic(nn.Module):
    def __init__(self, parameter_dim, loss_dim, action_dim):
        super(Critic, self).__init__()
        self.parameter_dim = parameter_dim
        self.loss_dim = loss_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(parameter_dim,action_dim)

        self.fc2 = nn.Linear(action_dim*(int(args.num_users*args.frac+3)), 1)

    def forward(self, parameters, last_loss, last_weight, action):
        parameter_lst = []
        for i in range(self.action_dim):
            parameter_lst.append(self.fc1(parameters[:,i,:]))
        parameter_layer = torch.cat(parameter_lst,dim=1)
        x = torch.cat([parameter_layer, last_loss, last_weight, action], dim=1)
        q = self.fc2(x)

        return q
    
    
    
GAMMA=0.99
class Trainer:
    def __init__(self, parameter_dim, loss_dim, action_dim, replay_buffer):
        self.parameter_dim = parameter_dim
        self.loss_dim = loss_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.iter = 0
        self.loss_critic_save = []
        self.loss_actor_save = []
        #self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.parameter_dim, 
                                 self.loss_dim, 
                                 self.action_dim).to(args.device)
        self.target_actor = Actor(self.parameter_dim, 
                                 self.loss_dim, 
                                 self.action_dim).to(args.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),0.01)

        self.critic = Critic(self.parameter_dim, 
                                 self.loss_dim, 
                                 self.action_dim).to(args.device)
        self.target_critic = Critic(self.parameter_dim, 
                                 self.loss_dim, 
                                 self.action_dim).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),0.01)

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)
                
    def get_exploitation_action(self, parameters, last_loss, last_weight):
        action = self.target_actor.forward(parameters, last_loss, last_weight).detach()
        return action.data.numpy()

#     def get_exploration_action(self, state):
#         """
#         gets the action from actor added with exploration noise
#         :param state: state (Numpy array)
#         :return: sampled action (Numpy array)
#         """
#         state = Variable(torch.from_numpy(state))
#         action = self.actor.forward(state).detach()
#         new_action = action.data.numpy() + (self.noise.sample())
#         return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        with torch.autograd.set_detect_anomaly(True):
            s1,a1,r1,s2 = self.replay_buffer.sample(40)
#                 print('s11:',s1[1])
#                 print('a1:',a1)
#             s1 = Variable(torch.from_numpy(np.array(s1)))
#             a1 = Variable(torch.from_numpy(np.array(a1)))
#             r1 = Variable(r1.float(), requires_grad=True)
            r1 = r1.to(args.device)
            r1 = Variable(r1)
            a1 = Variable(a1)
            for i in range(len(s1)):
                s1[i] = Variable(s1[i])
                s2[i] = Variable(s2[i])

#             s2 = Variable(torch.from_numpy(np.array(s2)))
            # ---------------------- optimize critic ----------------------
            # Use target actor exploitation policy here for loss evaluation
            # 这里应该是TD的方法
            a2 = self.target_actor.forward(s2[0],s2[1],s2[2]).detach()
            next_val = torch.squeeze(self.target_critic.forward(s2[0],s2[1],s2[2], a2).detach())
            y_expected = torch.squeeze(r1,dim=1) + GAMMA*next_val
            y_predicted = torch.squeeze(self.critic.forward(s1[0],s1[1],s1[2], a1))
#             print('grad:',s1[0].requires_grad, a1.requires_grad)
#             print('ye:',y_expected)
#             print('yp:',y_predicted)
            loss_critic = F.smooth_l1_loss(y_predicted.float(), y_expected.float())
            self.critic_optimizer.zero_grad()
#             loss_critic.backward(retain_graph=True)
            loss_critic.backward()
            self.critic_optimizer.step()
            self.loss_critic_save.append(loss_critic)
            # ---------------------- optimize actor ----------------------
            pred_a1 = self.actor.forward(s1[0],s1[1],s1[2])
            loss_actor = -1*torch.sum(self.critic.forward(s1[0],s1[1],s1[2], pred_a1))
            self.actor_optimizer.zero_grad()
#             loss_actor.backward(retain_graph=True)
            loss_actor.backward()
            self.actor_optimizer.step()
            self.loss_actor_save.append(loss_actor)
            
            self.soft_update(self.target_actor, self.actor, 0.001)
            self.soft_update(self.target_critic, self.critic, 0.001)

        # if self.iter % 100 == 0:
        #     print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
        #         ' Loss_critic :- ', loss_critic.data.numpy()
        # self.iter += 1

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        
        
        
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)
        
class QEEN(nn.Module):
    def __init__(self, input_dim, emb_feature_dim=100, dim_hidden=512, hidden_dim=100):
        super(QEEN, self).__init__()
        self.layer_input = nn.Linear(input_dim, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, emb_feature_dim)
        self.emb_feature_dim = emb_feature_dim
        layers = [
            nn.Linear(self.emb_feature_dim, hidden_dim),
        ]
        layers.append(nn.ReLU(inplace=True))
#         layers.append(
#             nn.Linear(hidden_dim,hidden_dim),
#         )
        self.mlp = nn.Sequential(*layers)
        self.ih1_weights = nn.Linear(2048, 4)
        self.ih1_bias = nn.Linear(2048, 1)
        self.hh1_weights = nn.Linear(2048, 512)
        self.hh1_bias = nn.Linear(2048, 1)
        self.ih2_weights = nn.Linear(2048, 512)
        self.ih2_bias = nn.Linear(2048, 1)
        self.hh2_weights = nn.Linear(2048, 512)
        self.hh2_bias = nn.Linear(2048, 1)
        self.fc_weights = nn.Linear(512, 1)
        self.fc_bias = nn.Linear(1, 1)
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


    def Reverse_raw_emb(self, x):
        features = self.mlp(x)
        output_features = torch.cat([
            self.ih1_weights(features),
            self.ih1_bias(features),
            self.hh1_weights(features),
            self.hh1_bias(features),
            self.ih2_weights(features),
            self.ih2_bias(features),
            self.hh2_weights(features),
            self.hh2_bias(features),
            self.fc_weights(features),
            self.fc_bias(features)
        ], 1)
        return output_features
