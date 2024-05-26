import os
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer3 import ReplayBuffer
from velodyne_env import GazeboEnv
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter



# linear [-0.5, 0, 0.5] ang [-1, 0, 1]

    
class CNN(nn.Module):
    def __init__(self, image_dim, state_dim, out_state_dim=200):
        super().__init__()  

        self.model1 = nn.Sequential(
            nn.Conv2d(image_dim[0], 3, 11, stride=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(5, stride=2),
            nn.Conv2d(3, 6, 3),
            nn.PReLU(),
            nn.MaxPool2d(5, stride=2),
            nn.Flatten()
        )
        
        self.l1 = nn.Sequential(
            nn.Linear(state_dim, 14 * 14 * 6),
            nn.PReLU()
        )

        self.l2 = nn.Sequential(
            nn.Linear(14 * 14 * 6, out_state_dim),
            nn.PReLU()
        )

        self.to(device)

    def forward(self, image, state):
          out1 = self.model1(image)
          out2 = self.l1(state)
          out = out1 + out2
          return self.l2(out)


class Agent(nn.Module):
    def __init__(self, image_size, state_dim):
        super().__init__()  

        self.cnn = CNN(image_size, state_dim, out_state_dim=100)
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.PReLU()
        )
        self.lv_model = nn.Linear(50, 5)
        self.av_model = nn.Linear(50, 5)
        self.to(device)

    def forward(self, image_state, env_state):
        features = self.cnn(image_state, env_state)
        out = self.model(features)
        lv_scores = F.softmax(self.lv_model(out), dim=-1)
        av_scores = F.softmax(self.av_model(out), dim=-1)
        return T.cat([lv_scores, av_scores], dim=1)
    

def create_state(state):
    image_state = T.tensor(state[0], dtype=T.float).to(device)
    env_state = T.tensor(state[1], dtype=T.float).to(device)
    return image_state, env_state

def get_action(q_values):
    scores = q_values.squeeze(0).reshape(-1, 2, 5).max(dim=-1)[1].squeeze(0)
    return [linear_actions[scores[0]], angular_actions[scores[1]]]

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += T.mean(T.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 
    

def train(cq, tq, eloss):
    global episode_reward
    if memory.mem_cntr < batch_size:
        return cq, tq, eloss
    state, reward, next_state, done = memory.sample_buffer(batch_size)
    reward = T.tensor(reward, dtype=T.float32, device=device)
    done = T.tensor(done, dtype=T.float32, device=device)
    i_state, e_state = create_state(state)
    current_q_values = agent(i_state, e_state)
    next_i_state, next_e_state = create_state(next_state)
    next_q_values = agent(next_i_state, next_e_state)
    max_next_q_l = T.max(next_q_values[:, :5], dim=-1)[0]
    max_next_q_a = T.max(next_q_values[:, 5:], dim=-1)[0]
    target_q_values = current_q_values.clone()
    l_action = current_q_values[:, :5].argmax(dim=1)
    a_action = current_q_values[:, 5:].argmax(dim=1)
    target_q_values[:, l_action] = reward + discount_factor * max_next_q_l * (1 - done)
    target_q_values[:, a_action] = reward + discount_factor * max_next_q_a * (1 - done)

    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cq += current_q_values.clone().cpu().detach().mean()
    tq += target_q_values.clone().cpu().detach().mean()
    eloss += loss.clone().cpu().detach().mean()
    episode_reward += reward.clone().cpu().detach().mean()

    return cq, tq, eloss



writer = SummaryWriter()

device = T.device("cuda")
state_dim = 5
image_size = (1, 224, 224)

agent = Agent(image_size=image_size, state_dim=state_dim)
init_weights(agent, "xavier")

batch_size = 16

memory = ReplayBuffer(batch_size+1, state_dim, image_size)

env = GazeboEnv("multi_robot_scenario.launch", 0)

optimizer = T.optim.Adam(agent.parameters(), lr=0.001)

linear_actions = [0, 0.3, 0.5, 0.8, 1]
angular_actions = [-1, -0.5,  0, 0.5, 1]

# # hyperparameters
num_episodes = 1000
max_steps_per_episode = 500
exploration_prob = [.8, .4] 
min_exploration_prob = 0.001
exploration_decay = [0.95, 0.995] 
discount_factor = 0.99

print_freq = 100

loss_fn = nn.MSELoss()
global episode_reward
episode_reward = 0

for episode in range(num_episodes):
    state = env.reset(episode)
    episode_reward = 0
    cq = tq = eloss = 0
    for step in range(max_steps_per_episode):
        with T.no_grad():
            i_state, e_state = create_state(state)
            q_values = agent(i_state.unsqueeze(0), e_state.unsqueeze(0))
            action = get_action(q_values)
            
            if np.random.rand() < exploration_prob[0]:
                action[0] = env.sample_action()[0]
            if np.random.rand() < exploration_prob[1]:
                action[1] = env.sample_action()[1]

        next_state, reward, done, _ = env.step(action)

        memory.store_transition(state, reward, next_state, done)

        cq, tq, eloss = train(cq, tq, eloss)
        
        state = next_state

        writer.add_scalar('Exploration Prob/linear', exploration_prob[0], step * (1 + episode))
        writer.add_scalar('Exploration Prob/Angular', exploration_prob[1], step * (1 + episode))
        if done:
            break

    exploration_prob[0] = max(min_exploration_prob, exploration_prob[0] * exploration_decay[0])
    exploration_prob[1] = max(min_exploration_prob, exploration_prob[1] * exploration_decay[1])

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    if (episode + 1) % 250 == 0:
        print(f"Save checkpoint at epoch {episode+1}")
        T.save(agent.state_dict(), f'model{episode+1}.pth')

    writer.add_scalar('Q_values/current', cq / max_steps_per_episode, episode)
    writer.add_scalar('Q_values/target', tq / max_steps_per_episode, episode)
    writer.add_scalar('Loss', eloss / max_steps_per_episode, episode)
    writer.add_scalar('Avg Reward', episode_reward, episode)
