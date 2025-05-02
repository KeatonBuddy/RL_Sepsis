import os
import numpy as np
import pyprind
import torch

class IQLExperiment(object):
    def __init__(self, data_loader_train, data_loader_validation, iql_network, ps, ns, folder_location, folder_name, saving_period, rng, resume):
        self.rng = rng
        self.data_loader_train = data_loader_train
        self.data_loader_validation = data_loader_validation
        self.iql_network = iql_network
        self.ps = ps  
        self.ns = ns  
        self.batch_num = 0
        self.saving_period = saving_period  
        self.resume = resume 
        storage_path = os.path.join(os.path.abspath(folder_location), folder_name)
        self.storage_rl = os.path.join(storage_path, 'iql_' + self.iql_network.sided_Q)
        self.checkpoint_folder = os.path.join(storage_path, 'iql_' + self.iql_network.sided_Q + '_checkpoints')
        if not os.path.exists(self.storage_rl):
            os.mkdir(self.storage_rl)
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        
    def do_epochs(self, number):
        if self.resume:
            checkpoint_file = os.path.join(self.checkpoint_folder, 'checkpoint_latest.pt')
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file)
                self.iql_network.resume(
                    q_network_state_dict=checkpoint['q_network_state_dict'],
                    v_network_state_dict=checkpoint['v_network_state_dict'],
                    target_q_network_state_dict=checkpoint['target_q_network_state_dict'],
                    q_optimizer_state_dict=checkpoint['q_optimizer_state_dict'],
                    v_optimizer_state_dict=checkpoint['v_optimizer_state_dict']
                )
                self.curr_epoch = checkpoint['epoch'] + 1
                self.all_epoch_steps = checkpoint['epoch_steps']
                self.all_epoch_validation_steps = checkpoint['epoch_validation_steps']
                self.all_epoch_loss = checkpoint['loss']
                self.all_epoch_validation_loss = checkpoint['validation_loss']
                print('Starting from epoch: {0} and continuing upto epoch {1}'.format(self.curr_epoch, number))
            else:
                print('No checkpoint found. Starting from scratch.')
                self.curr_epoch = 0
                self.all_epoch_steps = []
                self.all_epoch_validation_steps = []
                self.all_epoch_loss = []
                self.all_epoch_validation_loss = []
        else:
            self.curr_epoch = 0
            self.all_epoch_steps = []
            self.all_epoch_validation_steps = []
            self.all_epoch_loss = []
            self.all_epoch_validation_loss = []
            
        self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
        self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
        
        for epoch in range(self.curr_epoch, number):
            epoch_done = False
            epoch_steps = 0
            epoch_loss = 0

            bar = pyprind.ProgBar(self.data_loader_train.num_minibatches_epoch)
            while not epoch_done:
                bar.update()
                s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_train.get_next_minibatch()
                epoch_steps += len(s)
                loss = self.iql_network.learn(s, actions, rewards, next_s, terminals)
                epoch_loss += loss
                
            self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
            self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
            self.all_epoch_loss.append(epoch_loss/epoch_steps)
            self.all_epoch_steps.append(epoch_steps)
            
            if (epoch + 1) % self.saving_period == 0:
                self._do_eval()
                try:
                    torch.save({
                        'epoch': epoch,
                        'q_network_state_dict': self.iql_network.q_network.state_dict(),
                        'v_network_state_dict': self.iql_network.v_network.state_dict(),
                        'target_q_network_state_dict': self.iql_network.target_q_network.state_dict(),
                        'q_optimizer_state_dict': self.iql_network.q_optimizer.state_dict(),
                        'v_optimizer_state_dict': self.iql_network.v_optimizer.state_dict(),
                        'loss': self.all_epoch_loss,
                        'validation_loss': self.all_epoch_validation_loss,
                        'epoch_steps': self.all_epoch_steps,
                        'epoch_validation_steps': self.all_epoch_validation_steps,
                    }, os.path.join(self.checkpoint_folder, 'checkpoint' + str(epoch) +'.pt'))
                    
                    torch.save({
                        'epoch': epoch,
                        'q_network_state_dict': self.iql_network.q_network.state_dict(),
                        'v_network_state_dict': self.iql_network.v_network.state_dict(),
                        'target_q_network_state_dict': self.iql_network.target_q_network.state_dict(),
                        'q_optimizer_state_dict': self.iql_network.q_optimizer.state_dict(),
                        'v_optimizer_state_dict': self.iql_network.v_optimizer.state_dict(),
                        'loss': self.all_epoch_loss,
                        'validation_loss': self.all_epoch_validation_loss,
                        'epoch_steps': self.all_epoch_steps,
                        'epoch_validation_steps': self.all_epoch_validation_steps,
                    }, os.path.join(self.checkpoint_folder, 'checkpoint_latest.pt'))
                    
                    np.save(os.path.join(self.storage_rl, 'iql_losses.npy'), np.array(self.all_epoch_loss))
                    np.save(os.path.join(self.storage_rl, 'iql_validation_losses.npy'), np.array(self.all_epoch_validation_loss))
                except:
                    print(">>> Cannot save files. On Windows: the files might be open.")
        
    def _do_eval(self):
        epoch_val_steps = 0
        epoch_val_loss = 0
        epoch_done = False
        bar = pyprind.ProgBar(self.data_loader_validation.num_minibatches_epoch)
        while not epoch_done:
            bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_validation.get_next_minibatch()
            epoch_val_steps += len(s)
            loss = self.iql_network.get_loss(s, actions, rewards, next_s, terminals)
            epoch_val_loss += loss
        self.all_epoch_validation_loss.append(epoch_val_loss / epoch_val_steps)
        self.all_epoch_validation_steps.append(epoch_val_steps)