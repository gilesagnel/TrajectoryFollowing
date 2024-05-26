from train_v2 import Agent, create_state, get_action
import torch as T
from velodyne_env import GazeboEnv


def test_agent(agent, env, num_episodes, max_steps_per_episode):
    agent.eval()  
    avg_rewards = []
    for episode in range(num_episodes):
        state = env.reset(episode)
        episode_reward = 0
        for step in range(max_steps_per_episode):
            with T.no_grad():
                i_state, e_state = create_state(state)
                q_values = agent(i_state, e_state)
                action = get_action(q_values)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break
        
        avg_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = sum(avg_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")


device = T.device("cuda")
checkpoint_path = 'model1000.pth'
agent = Agent(image_size=(1, 224, 224), state_dim=5)  
agent.load_state_dict(T.load(checkpoint_path))
env = GazeboEnv("multi_robot_scenario.launch", 0)

num_episodes_test = 10  
test_agent(agent, env, num_episodes_test, 500)  
