import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

EPISODES = 5000
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
device = torch.device("cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

def one_hot_state(state, size):
    vec = np.zeros(size)
    vec[state] = 1.0
    return vec


memory = deque(maxlen=MEMORY_SIZE)

state_size = env.observation_space.n
action_size = env.action_space.n
q_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LR)


def train():
    epsilon = EPSILON_START
    reward_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_vec = one_hot_state(state, state_size)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_vec = one_hot_state(next_state, state_size)
            memory.append((state_vec, action, reward, next_state_vec, done))

            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.from_numpy(np.array(states)).float().to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.from_numpy(np.array(next_states)).float().to(device)
                dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)

                q_values = q_net(states_tensor).gather(1, actions_tensor)
                next_q_values = target_net(next_states_tensor).max(1)[0].detach().unsqueeze(1)
                targets = rewards_tensor + GAMMA * next_q_values * (~dones_tensor)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        reward_history.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    return reward_history


def optimal():
    state, _ = env.reset()
    done = False
    path = []

    while not done:
        path.append(state)
        state_vec = one_hot_state(state, state_size)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = q_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_vec = one_hot_state(next_state, state_size)
        memory.append((state_vec, action, reward, next_state_vec, done))

        state = next_state

    return path



reward_log = train()


window = 300
reward = np.convolve(reward_log, np.ones(window)/window, mode='valid')


plt.plot(reward)
plt.xlabel("Эпизоды")
plt.ylabel("Суммарное вознаграждение")
plt.title("DQN")
#plt.ylim(-10000, 100)
plt.show()


print(optimal())



