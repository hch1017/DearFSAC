import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
from torch.autograd import Variable

# Hyper Parameters
BATCH_SIZE = 40
LR = 0.01                   # learning rate
EPSILON = 0.95             # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency

class Net(nn.Module):

    def __init__(self, parameter_dim, loss_dim, args):
        super(Net, self).__init__()
        self.args = args
        self.parameter_dim = parameter_dim
        self.loss_dim = loss_dim

        self.fc1 = nn.Linear(parameter_dim, loss_dim)
        self.fc2 = nn.Linear(loss_dim*(int(args.num_users+1)),100) #输出100个Q

    def forward(self, parameters, last_loss):
        parameter_lst = []
        for i in range(self.loss_dim):
            parameter_lst.append(self.fc1(parameters[:,i,:]))
        parameter_layer = torch.cat(parameter_lst,dim=1)
        x = torch.cat([parameter_layer, last_loss],dim=1)
        q = self.fc2(x)

        return q

class DQN(object):
    def __init__(self, parameter_dim, loss_dim, replay_buffer, args):
        self.parameter_dim = parameter_dim #（batch, 10100）
        self.loss_dim = loss_dim
        self.replay_buffer = replay_buffer
        self.args = args
        self.iter = 0
        #self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.eval_net = Net(self.parameter_dim, self.loss_dim, args).to(args.device)
        self.target_net = Net(self.parameter_dim, self.loss_dim, args).to(args.device)

        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), 0.0001)
        self.loss_func = nn.MSELoss()
        self.loss_save = []
         
    #选q最大的local，直接赋值给global，只用一个local参与训练的目的是加快训练
    
    #选q最大的k个local,初始是k个，之后递减
    def choose_action_train(self, parameters, loss):
#         q = self.eval_net.forward(parameters)
# #         print('q',q)
#         action = torch.max(q, 1)[1]
# #         print('a', action)

#         #这是索引
#         # 1*1
#         return action.numpy().tolist()[0]

        self.eval_net.eval()
        q = self.eval_net.forward(parameters, loss)
#         int(self.args.frac * self.args.num_users)
#         print(q)
        topk = q.topk(self.args.k, dim=1, largest=True, sorted=True)
        values = topk[0].detach().numpy().tolist()
#         print(values)
        action = topk[1].detach().numpy().tolist()
#         print(action)
#         #l是q的列表形式
#         l = q.detach().numpy()[0][0].tolist()
        
#         #最大的k个q值
#         topk = heapq.nlargest(, l)
        
#         #找索引
#         action = []
#         for i in topk:
#             action.append(l.index(i))
        
        #算权值
#         topk_sum = np.sum(values)
#         for i in range(len(values)):
#             values[i] /= topk_sum
#         weight = values
        
        #softmax
        weight = F.softmax(torch.tensor(values), dim=1)
        return action[0], weight.squeeze(0)
    
    
    #选q最大的k个local
    def choose_action_run(self, parameters, loss):
        q = self.eval_net.forward(parameters, loss)
#         int(self.args.frac * self.args.num_users)
#         print(q)
        topk = q.topk(self.args.k_validation, dim=1, largest=True, sorted=True)
        values = topk[0].detach().numpy().tolist()
#         print(values)
        action = topk[1].detach().numpy().tolist()
#         print(action)
#         #l是q的列表形式
#         l = q.detach().numpy()[0][0].tolist()
        
#         #最大的k个q值
#         topk = heapq.nlargest(, l)
        
#         #找索引
#         action = []
#         for i in topk:
#             action.append(l.index(i))
        
        #算权值
#         topk_sum = np.sum(values)
#         for i in range(len(values)):
#             values[i] /= topk_sum
#         weight = values
        
        #softmax
        weight = F.softmax(torch.tensor(values), dim=1)
        return action[0], weight.numpy().tolist()[0]

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        self.eval_net.train()
        epoch = 3
        for i in range(epoch):
            s1,a1,r1,s2 = self.replay_buffer.sample(BATCH_SIZE)
#             s1,a1,r1,s2, indices, weights = self.replay_buffer.sample(BATCH_SIZE, beta)
    #         print(s1.shape)
    #         print(a1.shape)
#             print(s1[:2])
#             print(a1[:2])
#             print(r1[:2])
#             s1 = Variable(s1, requires_grad=True)
            r1 = Variable(r1.float(), requires_grad=True)
#             s2 = Variable(s2, requires_grad=True)
#             weights = Variable(torch.FloatTensor(weights))
    #         print('s1shape:',s1.shape)

            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # q_eval w.r.t the action in experience
            q_eval = self.eval_net.forward(s1[0], s1[1]) # BATCH_SIZE*100
#             print('a1:',a1[:2])
#             print('q_eval',q_eval[:2])
#             print('a_reshape', a1.reshape(-1,1)[:2])
#             q_eval = torch.gather(q_eval, 1, torch.LongTensor(a1.reshape(-1,1)))
            q_eval = torch.gather(q_eval, 1, a1)
    #         q_eval = torch.index_select(q_eval, dim=1, index=a1)#(batch_size, 1)
#             print('chosen_q_eval:', q_eval) #10*10

            q_next = self.target_net.forward(s2[0], s2[1]).detach()     # detach from graph, don't backpropagate
#             print('q_next',q_next[0])
#             print('type:', q_eval.type())

#             q_target = r1 + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)  # shape (batch, 1)
            q_target = r1 + GAMMA * q_next.topk(self.args.k, dim=1, largest=True, sorted=True)[0]
#             print('q_next',q_next.max(1)[0], q_next.max(1)[0].shape)
#             print('q_next_view:', q_next.max(1)[0].view(10, 1), q_next.max(1)[0].view(10, 1).shape)
#             print(r1)
#             print(r1.type())
#             print(q_target.type())
            
            loss = ((q_eval-q_target) ** 2)
#             prios = loss + 1e-5
            loss = torch.mean(loss)
#             loss = torch.mean(loss)
#             print('eval', torch.mean(q_eval[0]))
#             print('taregt', torch.mean(q_target[0]))
            print('loss', loss)
            self.loss_save.append(loss.tolist())
            self.optimizer.zero_grad()
            loss.backward()
#             self.replay_buffer.update_priorities(indices, prios.data.numpy())
            self.optimizer.step()
    #         self.soft_update(self.target_net, self.eval_net, 0.001) 