import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
from torch.autograd import Variable

# Hyper Parameters
BATCH_SIZE = 10
LR = 0.01                   # learning rate
EPSILON = 0.95             # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency

class Net(nn.Module):

    def __init__(self, parameter_dim, action_dim, args):
        super(Net, self).__init__()
        self.args = args
        self.parameter_dim = parameter_dim #10100
        self.action_dim = action_dim

        # N+1 即global的p拼接上所有local的p
        self.fc1 = nn.Linear(parameter_dim, 1000)
        self.fc2 = nn.Linear(1000, 100)

    def forward(self, parameters):
        #拼接的p
        x = self.fc1(parameters)
        q = self.fc2(x)
        
        #100维
        return q

class DQN(object):
    def __init__(self, parameter_dim, action_dim, replay_buffer, args):
        self.parameter_dim = parameter_dim #（batch, 10100）
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.args = args
        self.iter = 0
        #self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.eval_net = Net(self.parameter_dim, self.action_dim, args).to(args.device)
        self.target_net = Net(self.parameter_dim, self.action_dim, args).to(args.device)

        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), 0.0001)
        self.loss_func = nn.MSELoss()
        self.loss_save = []
        
        
    #选q最大的local，直接赋值给global，只用一个local参与训练的目的是加快训练
    def choose_action_train(self, parameters):
        self.eval_net.eval()
        q = self.eval_net.forward(parameters)

        topk = q.topk(self.args.k, dim=1, largest=True, sorted=True)
        values = topk[0].detach().numpy().tolist()
        action = topk[1].detach().numpy().tolist()

        return action[0]
    
    #选q最大的10个local，再将它们的q用softmax变成权值
    def choose_action_run(self, parameters):
        q = self.eval_net.forward(parameters)
        topk = q.topk(self.args.k_validation, dim=1, largest=True, sorted=True)
        values = topk[0].detach().numpy().tolist()
        action = topk[1].detach().numpy().tolist()
        
        print(action[0])
        return action[0]


    def optimize(self, beta):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        self.eval_net.train()
        epoch = 5
        for i in range(epoch):
#             s1,a1,r1,s2,indices,weights = self.replay_buffer.sample(BATCH_SIZE, beta)
            s1,a1,r1,s2 = self.replay_buffer.sample(BATCH_SIZE)
            s1 = Variable(s1, requires_grad=True)
            r1 = Variable(r1.float(), requires_grad=True)
            s2 = Variable(s2, requires_grad=True)
#             weights = Variable(torch.FloatTensor(weights))
    #         print('s1shape:',s1.shape)

            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # q_eval w.r.t the action in experience
            q_eval = self.eval_net.forward(s1) # 10*100
            q_eval = torch.gather(q_eval, 1, a1)

            q_next = self.eval_net.forward(s2).detach()
            a_next = q_next.topk(self.args.k, dim=1, largest=True, sorted=True)[1]

            q_target_next = self.target_net.forward(s2).detach()
            q_target_next = torch.gather(q_target_next, 1, a_next)

            q_target = r1 + GAMMA * q_target_next  # shape (batch, 1)

#             loss = self.loss_func(q_eval, q_target)
            loss = ((q_eval-q_target) ** 2)
#             prios = loss + 1e-5
            loss = torch.mean(loss)
            print('loss：', loss)
            
            self.loss_save.append(torch.mean(loss).tolist())
            
            self.optimizer.zero_grad()
            loss.backward()
#             self.replay_buffer.update_priorities(indices, prios.data.numpy())
            self.optimizer.step()