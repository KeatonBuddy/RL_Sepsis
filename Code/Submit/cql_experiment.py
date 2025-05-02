import os
import yaml
import numpy as np
import torch
import pyprind
from utils import DataLoader

class CQLExperiment(object):
    def __init__(self, data_loader_train, data_loader_validation, q_network, ps, ns, 
                 folder_location=None, folder_name=None, saving_period=20, rng=None, resume=False):

        self.train_loader = data_loader_train
        self.validation_loader = data_loader_validation
        self.q_network = q_network
        self.ps, self.ns = ps, ns
        self.saving_period = saving_period
        self.rng = rng

        self.folder_name = folder_name
        self.folder_location = folder_location
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
            
        self.storage = os.path.join(folder_location, self.folder_name)
        self.storage_rl = os.path.join(self.storage, 'rl_cql_{}'.format(self.q_network.sided_Q))
        self.current_epoch = 0
        
        if not os.path.exists(self.storage_rl):
            os.makedirs(self.storage_rl)
            
        self.resume = resume
        if self.resume:
            self.load_most_recent_checkpoint()
            
        self.reset_loaders()

    def reset_loaders(self):
        self.train_loader.reset_transitions()
        self.validation_loader.reset_transitions()

    def assign_reward(self, is_terminal, outcome, action):

        reward = 0
        if is_terminal:
            if outcome == 0:  
                reward = -self.ns
            else:  
                reward = self.ps
        return reward

    def do_epochs(self, number=1):
        print("Training for {} epochs...".format(number))
        
        train_results = []
        validation_results = []
        
        for epoch in range(1, number + 1):
            self.current_epoch += 1
            
            train_loss = self.do_epoch(self.train_loader, training=True)
            train_results.append(train_loss)
            validation_loss = self.do_epoch(self.validation_loader, training=False)
            validation_results.append(validation_loss)
            
            if self.current_epoch % self.saving_period == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self.save_results(train_results, validation_results)
        
        return train_results, validation_results

    def do_epoch(self, loader, training=True):
        losses = []
        loader.refresh_transitions()

        nb_minibatch = loader.number_of_minibatches()
        progress_bar = pyprind.ProgBar(nb_minibatch, title="CQL {} {}".format(
            self.q_network.sided_Q, "training" if training else "validation"), monitor=True)
        
        for _ in range(nb_minibatch):
            s, a, o, s2, t = loader.get_minibatch()
            
            # rewards based on outcomes
            r = np.zeros_like(o, dtype=np.float32)
            for i in range(len(t)):
                r[i] = self.assign_reward(t[i], o[i], a[i])
            
            if training:
                loss = self.q_network.learn(s, a, r, s2, t)
            else:
                loss = self.q_network.get_loss(s, a, r, s2, t)
                
            losses.append(loss)
            progress_bar.update()
            
        return np.mean(losses)

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'q_network_state_dict': self.q_network.network.state_dict(),
            'target_network_state_dict': self.q_network.target_network.state_dict(),
            'optimizer_state_dict': self.q_network.optimizer.state_dict()
        }
        
        checkpoint_path = os.path.join(self.storage_rl, 'checkpoint_{}.pt'.format(self.current_epoch))
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoint saved to {}".format(checkpoint_path))
        
        # Save network weights separately
        weights_path = os.path.join(self.storage_rl, 'weights_{}.pt'.format(self.current_epoch))
        self.q_network.dump_network(weights_path)

    def load_most_recent_checkpoint(self):
        """Load the most recent checkpoint if available"""
        checkpoints = [f for f in os.listdir(self.storage_rl) if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            print("No checkpoints found to resume from.")
            return
            
        # find latest checkpoint
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(self.storage_rl, latest_checkpoint)
        
        #Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.current_epoch = checkpoint['epoch']
        self.q_network.resume(
            checkpoint['q_network_state_dict'],
            checkpoint['target_network_state_dict'],
            checkpoint['optimizer_state_dict']
        )
        
        print("Resumed from checkpoint {} (epoch {})".format(latest_checkpoint, self.current_epoch))

    def save_results(self, train_results, validation_results):
        results = {
            'train_loss': train_results,
            'validation_loss': validation_results,
            'epochs': list(range(self.current_epoch - len(train_results) + 1, self.current_epoch + 1))
        }
        
        results_path = os.path.join(self.storage_rl, 'results.yml')
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
