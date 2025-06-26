import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cpu")

NUM_MAPS = 5
EPISODES = 1000
INNER_EPISODES = 20
GAMMA = 0.99
ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998
TARGET_UPDATE_FREQ = 20

BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 10000
MIN_REPLAY_SIZE = 1000  # чтобы сначала собрать достаточно опыта

state_size = 16
action_size = 4
context_size = NUM_MAPS
input_size = state_size + context_size

class ContextualQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

q_net = ContextualQ().to(device)
target_net = ContextualQ().to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=ALPHA)
loss_fn = nn.MSELoss()

maps = [generate_random_map(size=4, p=0.9) for _ in range(NUM_MAPS)]
reward_log = []
pass_log = []

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

def verify():
    cnt = 0
    for map_idx in range(NUM_MAPS):
        desc = maps[map_idx]
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)
        context = one_hot(map_idx, context_size)

        state, _ = env.reset()
        done = False

        while not done:
            state_vec = one_hot(state, state_size)
            input_vec = np.concatenate([state_vec, context])
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals = q_net(input_tensor)
                action = q_vals.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            if state == 15:
                cnt += 1
        env.close()
    pass_log.append(cnt / NUM_MAPS)
    print(f"Verification pass ratio: {cnt / NUM_MAPS:.3f}")

def train():
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        map_idx = random.randint(0, NUM_MAPS - 1)
        desc = maps[map_idx]
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)
        context = one_hot(map_idx, context_size)

        total_reward = 0

        for _ in range(INNER_EPISODES):
            state, _ = env.reset()
            done = False

            while not done:
                state_vec = one_hot(state, state_size)
                input_vec = np.concatenate([state_vec, context])
                input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(device)

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_vals = q_net(input_tensor)
                        action = q_vals.argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # reward -= 0.01  # можно попробовать убрать штраф

                next_state_vec = one_hot(next_state, state_size)
                next_input = np.concatenate([next_state_vec, context])

                # Сохраняем опыт в буфер
                replay_buffer.push(input_vec, action, reward, next_input, done)

                state = next_state
                total_reward += reward

                # Обучаемся, если достаточно данных
                if len(replay_buffer) > MIN_REPLAY_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                    states_tensor = torch.FloatTensor(states).to(device)
                    next_states_tensor = torch.FloatTensor(next_states).to(device)
                    actions_tensor = torch.LongTensor(actions).to(device)
                    rewards_tensor = torch.FloatTensor(rewards).to(device)
                    dones_tensor = torch.BoolTensor(dones).to(device)

                    # Вычисляем Q-values для текущих состояний
                    q_values = q_net(states_tensor)
                    q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                    # Вычисляем Q-values для следующих состояний
                    with torch.no_grad():
                        next_q_values = target_net(next_states_tensor).max(dim=1)[0]
                    targets = rewards_tensor + GAMMA * next_q_values * (~dones_tensor)

                    loss = loss_fn(q_values, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        verify()

        print(f"Episode {episode}, Reward: {(total_reward / INNER_EPISODES):.2f}, Epsilon: {epsilon:.3f}")
        reward_log.append(total_reward / INNER_EPISODES)
        env.close()

train()

window = 50
rewards = np.convolve(reward_log, np.ones(window)/window, mode='valid')
passes = np.convolve(pass_log, np.ones(window)/window, mode='valid')

plt.plot(rewards, label="Reward")
plt.plot(passes, label="Pass ratio")
plt.xlabel("Эпизоды")
plt.ylabel("Среднее значение")
plt.legend()
plt.title("Contextual Q-learning с Replay Buffer")
plt.show()