import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, parameter_dim, action_dim, args, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.args = args
        self.parameter_dim = parameter_dim
        self.loss_dim = loss_dim
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)
        self.device = args.device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(parameter_dim, action_dim)
        self.fc2 = nn.Linear(action_dim*(int(args.num_users*args.frac+2)), hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, parameters, last_loss, last_weight):
        parameter_lst = []
        for i in range(self.action_dim):
            parameter_lst.append(self.fc1(parameters[:,i,:]))
        parameter_layer = torch.cat(parameter_lst,dim=1)
        x = torch.cat([parameter_layer,last_loss, last_weight],dim=1)
        x = F.relu(self.fc2(x)) # 32
        mu = self.mu(x)

        log_std = F.relu(self.log_std_linear(x))
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32, args):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(Critic, self).__init__()
        self.args = args
        self.seed = torch.manual_seed(seed)
        self.device = args.device
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size,action_size)

        self.fc2 = nn.Linear(action_size*(int(args.num_users*args.frac+3)), hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, parameters, last_loss, last_weight, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        parameter_lst = []
        for i in range(self.action_size):
            parameter_lst.append(self.fc1(parameters[:,i,:]))
        parameter_layer = torch.cat(parameter_lst,dim=1)
        x = torch.cat([parameter_layer, last_loss, last_weight, action], dim=1)
        x = F.relu(self.fc2(x))
        q = F.relu(self.fc3(x))

        return q


class DeepActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, device, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DeepActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        in_dim = hidden_size+state_size

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(in_dim, hidden_size)
        self.fc3 = nn.Linear(in_dim, hidden_size)
        self.fc4 = nn.Linear(in_dim, hidden_size)


        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        #self.reset_parameters() # check if this improves training

    def reset_parameters(self, init_w=3e-3):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.tensor)-> (float, float):

        x = F.relu(self.fc1(state))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc4(x))  

        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        #state = torch.FloatTensor(state).to(device) #.unsqzeeze(0)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()

class DeepCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, device, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(DeepCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        in_dim = hidden_size+action_size+state_size
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(in_dim, hidden_size)
        self.fc3 = nn.Linear(in_dim, hidden_size)
        self.fc4 = nn.Linear(in_dim, hidden_size)
        
        self.fc5 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xu = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(xu))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc4(x))

        return self.fc5(x)


class IQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, N, device="cuda:0"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device

        # Network Architecture

        self.head = nn.Linear(self.action_size+self.input_shape, layer_size) 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)    
        #weight_init([self.head_1, self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=32):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, action, num_tau=32):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]

        x = torch.cat((input, action), dim=1)
        x = torch.relu(self.head(x  ))
        
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)  #batch_size*num_tau, self.cos_layer_out
        # Following reshape and transpose is done to bring the action in the same shape as batch*tau:
        # first 32 entries are tau for each action -> thats why each action one needs to be repeated 32 times 
        # x = [[tau1   action = [[a1
        #       tau1              a1   
        #        ..               ..
        #       tau2              a2
        #       tau2              a2
        #       ..]]              ..]]  
        #action = action.repeat(num_tau,1).reshape(num_tau,batch_size*self.action_size).transpose(0,1).reshape(batch_size*num_tau,self.action_size)
        #x = torch.cat((x,action),dim=1)
        x = torch.relu(self.ff_1(x))

        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, 1), taus
    
    def get_qvalues(self, inputs, action):
        quantiles, _ = self.forward(inputs, action, self.N)
        actions = quantiles.mean(dim=1)
        return actions  



class DeepIQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, N, device="cuda:0"):
        super(DeepIQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.input_dim = action_size+state_size+layer_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device

        # Network Architecture

        self.head = nn.Linear(self.action_size+self.input_shape, layer_size) 
        self.ff_1 = nn.Linear(self.input_dim, layer_size)
        self.ff_2 = nn.Linear(self.input_dim, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_3 = nn.Linear(self.input_dim, layer_size)
        self.ff_4 = nn.Linear(self.layer_size, 1)    
        #weight_init([self.head_1, self.ff_1])  

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=32):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, action, num_tau=32):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        xs = torch.cat((input, action), dim=1)
        x = torch.relu(self.head(xs))
        x = torch.cat((x, xs), dim=1)
        x = torch.relu(self.ff_1(x))   
        x = torch.cat((x, xs), dim=1)
        x = torch.relu(self.ff_2(x))

        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)  #batch_size*num_tau, self.cos_layer_out
        # Following reshape and transpose is done to bring the action in the same shape as batch*tau:
        # first 32 entries are tau for each action -> thats why each action one needs to be repeated 32 times 
        # x = [[tau1   action = [[a1
        #       tau1              a1   
        #        ..               ..
        #       tau2              a2
        #       tau2              a2
        #       ..]]              ..]]  
        action = action.repeat(num_tau,1).reshape(num_tau,batch_size*self.action_size).transpose(0,1).reshape(batch_size*num_tau,self.action_size)
        state = input.repeat(num_tau,1).reshape(num_tau,batch_size*self.input_shape).transpose(0,1).reshape(batch_size*num_tau,self.input_shape)
        
        x = torch.cat((x,action,state),dim=1)
        x = torch.relu(self.ff_3(x))

        out = self.ff_4(x)
        
        return out.view(batch_size, num_tau, 1), taus
    
    def get_qvalues(self, inputs, action):
        quantiles, _ = self.forward(inputs, action, self.N)
        actions = quantiles.mean(dim=1)
        return actions  