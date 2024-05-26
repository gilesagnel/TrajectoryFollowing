import torch as T  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer():

    def __init__(self, max_size, env_dims=2, image_shape=(1, 224, 224)):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.image_state_memory = np.zeros((self.mem_size, *image_shape))
        self.env_state_memory = np.zeros((self.mem_size, env_dims))        
        self.new_image_state_memory = np.zeros((self.mem_size, *image_shape))
        self.new_env_state_memory = np.zeros((self.mem_size, env_dims))    
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)   
    
    def store_transition(self, state, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.image_state_memory[index] = state[0]
        self.env_state_memory[index] = state[1]
        self.new_image_state_memory[index] = state_[0]
        self.new_env_state_memory[index] = state_[1]
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)   
        batch = np.random.choice(max_mem, batch_size)
        states = [self.image_state_memory[batch], self.env_state_memory[batch_size]]        
        states_ = [self.new_image_state_memory[batch], self.new_env_state_memory[batch_size]]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]    

        return states, rewards, states_, dones