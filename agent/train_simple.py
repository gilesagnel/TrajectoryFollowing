import os
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer2 import ReplayBuffer
from velodyne_env import GazeboEnv


class Critic(nn.Module):
    def __init__(self, beta, input_dim=200):
        super(Critic, self).__init__()
        self.input_dims = input_dim    
        self.fc1_dims = 128    
        self.fc2_dims = 128    
        self.n_actions = 2     
        self.fc1 = nn.Linear(input_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.n_actions, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        out1 = F.relu(self.fc1(state))
        out2 = F.relu(self.fc2(action))
        out = out1 + out2
        q1 = self.q1(out) 
        return q1
    
class Actor(nn.Module):
    def __init__(self, alpha, input_dims):
        super(Actor, self).__init__()
        self.input_dims = 2
        self.fc1_dims = 128
        self.fc2_dims = 128
        self.n_actions = 2
        self.fc1 = nn.Linear(input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
    
    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        return T.sigmoid(self.mu(out))

    
class CNN(nn.Module):
    def __init__(self, image_dim, state_dim, out_state_dim=200):
          super().__init__()  

          self.conv1 = T.nn.Conv2d(image_dim[0], 3, 11, stride=3, padding=1)
          self.max1 = T.nn.MaxPool2d(5, stride=2)
          self.conv2 = T.nn.Conv2d(3, 6, 3)
          self.max2 = T.nn.MaxPool2d(5, stride=2)
          
          self.l1 = T.nn.Linear(state_dim, 14 * 14 * 6)

          self.l2 = T.nn.Linear(14 * 14 * 6, out_state_dim)
          self.to(device)

    def forward(self, image, state):
          out1 = F.relu(self.conv1(image))
          out1 = self.max1(out1)
          out1 = F.relu(self.conv2(out1))
          out1 = self.max2(out1)
          out1 = out1.reshape(out1.shape[0], -1)

          out2 = F.relu(self.l1(state))
          
          out = out1 + out2
          return F.relu(self.l2(out))


class Agent(object):
    def __init__(self, alpha, beta, tau, image_size, state_dim, feature_dim=200,
                  gamma=0.99, n_actions=2, max_size=1000000,  batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(batch_size+1, state_dim, n_actions, image_size)
        self.batch_size = batch_size
        self.cnn = CNN(image_size, state_dim)
        self.actor = Actor(alpha, feature_dim)
        self.critic = Critic(beta)
        self.target_actor = Actor(alpha, feature_dim)
        self.target_critic = Critic(beta)
        
        self.scale = 1.0
        self.noise = np.random.uniform(size=(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, state):
        image_state = T.tensor(state[0], dtype=T.float).to(device)
        image_state = image_state.unsqueeze(0)
        env_state = T.tensor(state[1], dtype=T.float).to(device)
        env_state = env_state.unsqueeze(0)
        self.cnn.eval()
        self.actor.eval()
        
        features = self.cnn(image_state, env_state)
        action = self.actor(features).squeeze(0)
        self.cnn.train()
        self.actor.train()
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(device)
        done = T.tensor(done).to(device)
        new_image_state = T.tensor(new_state[0], dtype=T.float).to(device)
        new_env_state = T.tensor(new_state[1], dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)
        image_state = T.tensor(state[0], dtype=T.float).to(device)
        env_state = T.tensor(state[1], dtype=T.float).to(device)

        self.cnn.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        ns_features = self.cnn(new_image_state, new_env_state)
        s_features = self.cnn(image_state, env_state)
        target_actions = self.target_actor(ns_features)
        critic_value_ = self.target_critic(ns_features, target_actions)
        critic_value = self.critic(s_features, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.cnn.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        print("loss", critic_loss.data, "target", target.data.mean(), "current", critic_value.data.mean())
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        s_features = self.cnn(image_state, env_state)
        mu = self.actor.forward(s_features)
        self.actor.train()
        actor_loss = -self.critic(s_features, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


device = T.device("cuda")

agent = Agent(alpha=0.000025, beta=0.00025, tau=0.001, image_size=(1, 224, 224),
              state_dim=5, batch_size=64, n_actions=2)

env = GazeboEnv("multi_robot_scenario.launch", 0)

expl_noise = 1  
expl_decay_steps = (
    500000  
)
expl_min = 0.1  

score_history = []
for i in range(10000):
    state = env.reset()
    done = False
    score = 0
    while not done:
        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

        action = agent.choose_action(state[:])
        if np.random.uniform() < 0.5:
            action[0] += np.random.normal(0, expl_noise)
        if np.random.uniform() < 0.1:
            action[1] = (action[1] * 2 -1) + np.random.normal(0, expl_noise)
        else:
            action[1] = 0
        action[0] = action[0].clip(0, 1)
        action[1] = action[1].clip(-1, 1)

        new_state, reward, done, target = env.step(action)
        agent.remember(state, action, reward, new_state, int(done))

        agent.learn()
        score += reward
        state = new_state
        
    score_history.append(score)