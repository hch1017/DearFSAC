from .replaybuffer import ReplayBuffer, PrioritizedReplay
from .SAC_network import Actor, Critic, DeepActor, DeepCritic, IQN, DeepIQN
import torch.optim as optim
import random
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
import torch
import numpy as np


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, args, action_prior="uniform"):
        self.iter = 0
        self.loss_critic_save = []
        self.loss_actor_save = []
        self.loss_Q_save = []
        
        self.args = args
        self.state_size = state_size
        self.action_size = action_size
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        random_seed = args.seed
        self.device = args.device
        self.per = args.per
        self.ere = args.ere
        self.n_step = args.n_step
        self.munchausen = args.munchausen
        self.distributional = args.distributional
        self.N = 32
        self.D2RL = args.d2rl
        self.m_alpha = 0.9
        self.m_tau = 0.03
        self.lo = -1
        self.batch_size = args.batch_size
        self.n_updates = args.n_updates
        self.buffer_size = int(args.replay_memory)
        self.gamma = args.gamma
        self.worker = args.worker
        self.tau = args.tau
        hidden_size = args.layer_size

        self.target_entropy = -action_size  # -dim(A)

        self.FIXED_ALPHA = args.alpha
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr_a) 
        self._action_prior = "uniform"
        
        # Actor Network 
        if self.D2RL:
            self.actor_local = DeepActor(state_size, action_size, random_seed, device, hidden_size).to(device)
        else:
            self.actor_local = Actor(state_size, action_size, random_seed, device, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=args.lr_a)     
        
        # Critic Network (w/ Target Network)
        if self.distributional:
            if self.D2RL:
                self.critic1 = DeepIQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic2 = DeepIQN(state_size, action_size, hidden_size, random_seed+1, self.N,  device).to(device)
                
                self.critic1_target = DeepIQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic1_target.load_state_dict(self.critic1.state_dict())

                self.critic2_target = DeepIQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic2_target.load_state_dict(self.critic2.state_dict())
            else:
                self.critic1 = IQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic2 = IQN(state_size, action_size, hidden_size, random_seed+1, self.N, device).to(device)
                
                self.critic1_target = IQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic1_target.load_state_dict(self.critic1.state_dict())

                self.critic2_target = IQN(state_size, action_size, hidden_size, random_seed, self.N, device).to(device)
                self.critic2_target.load_state_dict(self.critic2.state_dict())
        else: 

            if self.D2RL:
                self.critic1 = DeepCritic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic2 = DeepCritic(state_size, action_size, random_seed+1, device, hidden_size).to(device)
                
                self.critic1_target = DeepCritic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic1_target.load_state_dict(self.critic1.state_dict())

                self.critic2_target = DeepCritic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic2_target.load_state_dict(self.critic2.state_dict())
            else:
                self.critic1 = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic2 = Critic(state_size, action_size, random_seed+1, device, hidden_size).to(device)
                
                self.critic1_target = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic1_target.load_state_dict(self.critic1.state_dict())

                self.critic2_target = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
                self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr_c, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr_c, weight_decay=0) 
        print(self.actor_local)
        print(self.critic1)

        # Replay memory
        if self.per == 1:
            self.memory = PrioritizedReplay(self.buffer_size,
                                            self.batch_size,
                                            self.device, 
                                            seed=random_seed,
                                            gamma=self.gamma,
                                            ere=self.ere,
                                            n_step=self.n_step,
                                            parallel_env=self.worker,
                                            beta_frames=args.frames)
            self.learn = self.learn_per
        else:
            self.per = 0
            self.memory = ReplayBuffer(self.buffer_size,
                                       self.batch_size,
                                       self.device,
                                       random_seed,
                                       self.gamma,
                                       n_step=self.n_step,
                                       parallel_env=self.worker,
                                       ere=self.ere)
            if self.distributional:
                self.learn = self.learn_distr
            else:
                self.learn = self.learn_
        print("Using PER: {}".format(self.per))        
        print("Using Munchausen RL: {}".format(self.munchausen))
        print("Using N-step size: {}".format(self.n_step))



    def step(self, state, action, reward, next_state, step, ERE=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state)
        if ERE == False:
            # Learn, if enough samples are available in memory
            for _ in range(self.n_updates):
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(step, experiences, self.gamma)

    def ere_step(self, c_k):
        # Learn, if enough samples are available in memory
        experiences = self.memory.sample(c_k)
        self.learn(1, experiences, self.gamma)
            
    
    def act(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha, weights=1):
        actions_pred, log_pis = self.actor_local.evaluate(states)
        # Compute actor loss
        if self._action_prior == "normal":
            policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
            policy_prior_log_probs = policy_prior.log_prob(actions_pred)
        elif self._action_prior == "uniform":
            policy_prior_log_probs = 0.0

        q1 = self.critic1(states, actions_pred.squeeze(0))   
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q - policy_prior_log_probs)*weights).mean()
        return actor_loss, log_pis

    def learn_(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
            Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

            # take the mean of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if not self.munchausen:
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)
                    Q_targets = rewards.cpu() + (gamma**self.n_step * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu())) 
                else:
                    Q_targets = rewards.cpu() + (gamma**self.n_step * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.cpu())) 
            else:
                mu_m, log_std_m = self.actor_local(states)
                std = log_std_m.exp()
                dist = Normal(mu_m, std)
                log_pi_a = self.m_tau*dist.log_prob(actions).mean(1).unsqueeze(1).cpu()
                assert log_pi_a.shape == (self.batch_size, 1)
                munchausen_reward = (rewards.cpu() + self.m_alpha*torch.clamp(log_pi_a, min=self.lo, max=0))
                assert munchausen_reward.shape == (self.batch_size, 1)
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)
                    Q_targets = munchausen_reward + (gamma**self.n_step * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu())) 
                else:
                    Q_targets = munchausen_reward + (gamma**self.n_step * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.cpu())) 

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        assert Q_1.shape == Q_targets.shape, "Exp: {}  -- Target: {}".format(Q_1.shape, Q_targets.shape)
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets)
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets)
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if self.FIXED_ALPHA == None:
                actor_loss, log_pis = self.calc_policy_loss(states, self.alpha)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Compute alpha loss
                alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().detach()

            else:
                actor_loss, _ = self.calc_policy_loss(states, self.FIXED_ALPHA)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

    def learn_per(self, step, experiences, gamma, d=1):
            """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
            Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
            Critic_loss = MSE(Q, Q_target)
            Actor_loss = α * log_pi(a|s) - Q(s,a)
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            with torch.no_grad():
                next_action, log_pis_next = self.actor_local.evaluate(next_states)
                Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
                Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

                Q_target_next = torch.min(Q_target1_next, Q_target2_next)
                if not self.munchausen:
                    if self.FIXED_ALPHA == None:
                        # Compute Q targets for current states (y_i)
                        Q_targets = rewards.cpu() + (gamma * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
                    else:
                        Q_targets = rewards.cpu() + (gamma * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.cpu()))
                else:
                    mu_m, log_std_m = self.actor_local(states)
                    std = log_std_m.exp()
                    dist = Normal(mu_m, std)
                    log_pi_a = dist.log_prob(actions).mean(1).unsqueeze(1).cpu()
                    assert log_pi_a.shape == (self.batch_size, 1)
                    munchausen_reward = (rewards.cpu() + self.m_alpha*torch.clamp(self.m_tau*log_pi_a, min=self.lo, max=0))
                    assert munchausen_reward.shape == (self.batch_size, 1)
                    if self.FIXED_ALPHA == None:
                        # Compute Q targets for current states (y_i)
                        Q_targets = munchausen_reward + (gamma * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
                    else:
                        Q_targets = munchausen_reward + (gamma * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next .cpu()))
                
            # Compute critic loss
            Q_1 = self.critic1(states, actions).cpu()
            Q_2 = self.critic2(states, actions).cpu()
            
            td_error1 = Q_targets-Q_1
            td_error2 = Q_targets-Q_2
            critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
            critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
            prios = abs((torch.min(td_error1, td_error2)+1e-5)).squeeze().detach()
            #prios = abs((torch.cat((td_error1, td_error2)).mean(1) + 1e-5).squeeze()).detach()
            #prios = abs(td_error1 + 1e-5).squeeze().detach()
            # Update critics
            # critic 1
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            # critic 2
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            self.memory.update_priorities(idx, prios.data.cpu().numpy())
            if step % d == 0:
            # ---------------------------- update actor ---------------------------- #
                actor_loss, log_pis = self.calc_policy_loss(states, self.alpha, weights=weights)
                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                alpha_loss = (- (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu())*weights).mean() 
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().detach()

                # ----------------------- update target networks ----------------------- #
                self.soft_update(self.critic1, self.critic1_target)
                self.soft_update(self.critic2, self.critic2_target)


    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def learn_distr(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            Q_target1_next, _ = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
            Q_target2_next, _ = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

            # take the min of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next).transpose(1,2)
            if not self.munchausen:
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)  
                    Q_targets = rewards.cpu().unsqueeze(-1) + (gamma * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu().unsqueeze(-1))) 
                else:
                    Q_targets = rewards.cpu().unsqueeze(-1) + (gamma * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.cpu().unsqueeze(-1))) 
            else:
                mu_m, log_std_m = self.actor_local(states)
                std = log_std_m.exp()
                dist = Normal(mu_m, std)
                log_pi_a = self.m_tau*dist.log_prob(actions).mean(1).unsqueeze(1).cpu()
                assert log_pi_a.shape == (self.batch_size, 1)
                munchausen_reward = (rewards.cpu() + self.m_alpha*torch.clamp(log_pi_a, min=self.lo, max=0)).unsqueeze(-1)
                assert munchausen_reward.shape == (self.batch_size, 1, 1)
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)
                    Q_targets = munchausen_reward + (gamma * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu().unsqueeze(-1))) 
                else:
                    Q_targets = munchausen_reward + (gamma * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.cpu().unsqueeze(-1))) 

        # Compute critic loss
        Q_1, taus1 = self.critic1(states, actions, self.N)
        Q_2, taus2 = self.critic2(states, actions, self.N)
        assert Q_targets.shape == (self.batch_size, 1, self.N), "have shape: {}".format(Q_targets.shape)
        assert Q_1.shape == (self.batch_size, self.N, 1)

        # Quantile Huber loss
        td_error1 = Q_targets - Q_1.cpu()
        td_error2 = Q_targets - Q_2.cpu()

        assert td_error1.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l_1 = calculate_huber_loss(td_error1, 1.0)
        huber_l_2 = calculate_huber_loss(td_error2, 1.0)
        
        quantil_l_1 = abs(taus1.cpu() -(td_error1.detach() < 0).float()) * huber_l_1 / 1.0
        quantil_l_2 = abs(taus2.cpu() -(td_error2.detach() < 0).float()) * huber_l_2 / 1.0
        critic1_loss = quantil_l_1.sum(dim=1).mean(dim=1).mean()
        critic2_loss = quantil_l_2.sum(dim=1).mean(dim=1).mean()

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if self.FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha.detach()
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                q1 = self.critic1.get_qvalues(states, actions_pred.squeeze(0))   
                q2 = self.critic2.get_qvalues(states, actions_pred.squeeze(0))
                min_Q = torch.min(q1,q2).cpu()
                actor_loss = (alpha.detach() * log_pis.cpu() - min_Q - policy_prior_log_probs).mean()
                
            else:
                
                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                
                q1 = self.critic1.get_qvalues(states, actions_pred.squeeze(0))   
                q2 = self.critic2.get_qvalues(states, actions_pred.squeeze(0))
                min_Q = torch.min(q1,q2).cpu()
                actor_loss = (self.FIXED_ALPHA * log_pis.cpu() - min_Q - policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss