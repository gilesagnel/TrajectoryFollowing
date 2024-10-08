import numpy as np
import torch as T
import torch.nn as nn

from velodyne_env import GazeboEnv

from torch.utils.tensorboard import SummaryWriter
from models.agent import Agent
from models.replay_buffer3 import ReplayBuffer
from models.network import *
import random
    

# # Prepare states
# i_state, e_state = create_state(state)
# next_i_state, next_e_state = create_state(next_state)
# # Forward pass
# current_q_values = agent(i_state, e_state)
# next_q_values = agent(next_i_state, next_e_state)
# # Compute max next Q-values
# max_next_q_l = torch.max(next_q_values[:, :5], dim=-1)[0]
# max_next_q_a = torch.max(next_q_values[:, 5:], dim=-1)[0]
# # Clone current Q-values for target
# target_q_values = current_q_values.clone()
# # Compute target Q-values
# l_action = current_q_values[:, :5].argmax(dim=1)
# a_action = current_q_values[:, 5:].argmax(dim=1)
# target_q_values[:, l_action] = reward + discount_factor * max_next_q_l * (1 - done)
# target_q_values[:, a_action + 5] = reward + discount_factor * max_next_q_a * (1 - done)
# # Compute loss
# loss = loss_fn(current_q_values, target_q_values)
# # Backpropagation
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# # Monitor
# cq += current_q_values.mean().item()
# tq += target_q_values.mean().item()
# eloss += loss.item()
# return cq, tq, eloss


def train(cq, tq, eloss):
    global episode_reward
    if memory.mem_cntr < batch_size:
        return cq, tq, eloss
    state, reward, next_state, done = memory.sample_buffer(batch_size)
    reward = T.tensor(reward, dtype=T.float32, device=device)
    done = T.tensor(done, dtype=T.float32, device=device)
    i_state, e_state = create_state(state)
    next_i_state, next_e_state = create_state(next_state)

    current_q_values = agent(i_state, e_state)
    next_q_values = agent(next_i_state, next_e_state)
    
    max_next_q_l = T.max(next_q_values[:, :5], dim=-1)[0]
    max_next_q_a = T.max(next_q_values[:, 5:], dim=-1)[0]
    
    l_action = current_q_values[:, :5].argmax(dim=1)
    a_action = current_q_values[:, 5:].argmax(dim=1)
    
    target_q_values = current_q_values.clone()
    target_q_values[:, l_action] = reward + discount_factor * max_next_q_l * (1 - done)
    target_q_values[:, a_action + 5] = reward + discount_factor * max_next_q_a * (1 - done)

    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # diagnose_network(agent, "agent")
    optimizer.step()

    cq += current_q_values.clone().cpu().detach().mean()
    tq += target_q_values.clone().cpu().detach().mean()
    eloss += loss.clone().cpu().detach().mean()

    return cq, tq, eloss

def get_max_step(min_step, max_step, episode, max_episode):
    slope = (max_step - min_step) / max_episode
    interpolated_max_step = min_step + slope * episode
    interpolated_max_step = min(interpolated_max_step, max_step)
    return int(interpolated_max_step) 

writer = SummaryWriter()

device = T.device("cuda")
state_dim = 5
image_size = (3, 224, 224)

agent = Agent(image_size=image_size, state_dim=state_dim)
init_weights(agent, "xavier")

batch_size = 50

memory = ReplayBuffer(batch_size+1, state_dim, image_size)

linear_actions = [0, 0.3, 0.5, 0.8, 1]
angular_actions = [-0.5, -0.2,  0, 0.2, 0.5]

env = GazeboEnv("multi_robot_scenario.launch", action_space=[linear_actions, angular_actions])

optimizer = T.optim.Adam(agent.parameters(), lr=1e-4)


# # hyperparameters
num_episodes = 4000
max_steps_per_episode = 1000
exploration_prob = 1.0
min_exploration_prob = 0.01
exploration_decay = 0.9995
discount_factor = 0.99

print_freq = 100

loss_fn = nn.MSELoss()
episode_rewards = []
min_exploration_steps = [1]
exploration_step = 1

for episode in range(1, num_episodes):
    state = env.reset(episode)
    memory.empty()
    cq = tq = eloss = 0
    episode_reward = 0
    for step in range(max_steps_per_episode):
        agent.eval()
        if np.random.rand() < exploration_prob and step > get_max_step(10, max_steps_per_episode, episode, num_episodes//2):
            exploration_step += 1
            action = env.sample_action()
        else:
            with T.no_grad():
                i_state, e_state = create_state(state) 
                q_values = agent(i_state.unsqueeze(0), e_state.unsqueeze(0))
                action = env.get_action(q_values)

        agent.train()
        next_state, reward, done, target = env.step(action)

        episode_reward += reward

        memory.store_transition(state, reward, next_state, done)

        cq, tq, eloss = train(cq, tq, eloss)
        
        state = next_state
        if done and target:
            print(f"Episode {episode} was reached the goal")

        if done:
            break

    episode_rewards.append(episode_reward)        
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
    

    if episode % 10 == 0:
        avg_eps_rew = sum(episode_rewards) / 10
        print(f"Episode {episode}: Reward = {avg_eps_rew}")
        episode_rewards = []
    
    if (episode) % 250 == 0:
        save_model(agent, episode)

    writer.add_scalar('Q_values/current', cq / max_steps_per_episode, episode)
    writer.add_scalar('Q_values/target', tq / max_steps_per_episode, episode)
    writer.add_scalar('Loss', eloss / max_steps_per_episode, episode)
    writer.add_scalar('Avg Reward', episode_reward, episode)
    writer.add_scalar('Exploration Prob', exploration_prob, episode)

# env.destroy()
