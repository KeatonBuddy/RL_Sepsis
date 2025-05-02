import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl import QNetwork_64, QNetwork_128, QNetwork_6464, init_weights


class CQL(object):
    def __init__(self, state_dim, nb_actions, gamma,learning_rate, update_freq, use_ddqn,
                 rng, device, sided_Q, network_size, cql_alpha = 0.3, cql_tau = 5.0):
        
        self.rng = rng
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.update_counter = 0
        self.use_ddqn = use_ddqn
        self.device = device
        self.network_size = network_size
        self.cql_alpha = cql_alpha  
        self.cql_tau = cql_tau 
        
        # Initialize networks
        if self.network_size == 'small':
            QNetwork = QNetwork_64
        elif self.network_size == 'large':
            QNetwork = QNetwork_128
        elif self.network_size == '2layered':
            QNetwork = QNetwork_6464
            
        self.network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.target_network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
        self.sided_Q = sided_Q

    def train_on_batch(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)
        
        # Standard Q-learning loss
        q = self.network(s)
        q2 = self.target_network(s2).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)
        
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
            
        if self.sided_Q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1 - t)
        elif self.sided_Q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)
        elif self.sided_Q == 'both':
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
        
        q_loss = F.smooth_l1_loss(q_pred, bellman_target) # Standard Bellman error
        q_logsumexp = torch.logsumexp(q / self.cql_tau, dim=1) * self.cql_tau
        cql_reg = (q_logsumexp - q_pred).mean()
        
        #Total loss 
        loss = q_loss + self.cql_alpha * cql_reg
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().numpy()

    def get_loss(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        with torch.no_grad():
            q = self.network(s).detach()
            q2 = self.target_network(s2).detach()
            
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
            
        if self.sided_Q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1 - t)
        elif self.sided_Q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)
        elif self.sided_Q == 'both':
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
        
        q_loss = F.smooth_l1_loss(q_pred, bellman_target)
        
        # CQL Loss component
        q_logsumexp = torch.logsumexp(q / self.cql_tau, dim=1) * self.cql_tau
        cql_reg = (q_logsumexp - q_pred).mean()
        
        # Totall loss
        loss = q_loss + self.cql_alpha * cql_reg
        
        return loss.detach().cpu().numpy()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device)
        return self.network(s).detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        q = self.network(s).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states):
        return self.get_max_action(states)

    def learn(self, s, a, r, s2, term):
        """ Learning from one minibatch """
        loss = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return loss

    def dump_network(self, weights_file_path):
        try:
            torch.save(self.network.state_dict(), weights_file_path)
        except:
            pass

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def resume(self, network_state_dict, target_network_state_dict, optimizer_state_dict):
        self.network.load_state_dict(network_state_dict)
        self.target_network.load_state_dict(target_network_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        return _dict
