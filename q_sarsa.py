import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("CliffWalking-v0")

alpha = 0.1
gamma = 0.99
epsilon = 0
episodes = 500

def epsilon_greedy(Q, state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

def train(agent_type="q_learning"):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state)
        total_reward = 0

        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = epsilon_greedy(Q, next_state)

            if agent_type == "q_learning":
                target = reward + gamma * np.max(Q[next_state])
            elif agent_type == "sarsa":
                target = reward + gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (target - Q[state][action])

            state, action = next_state, next_action
            total_reward += reward

        rewards.append(total_reward)
    return rewards, Q


def optimal(Q):
    path = []
    state, _ = env.reset()
    action = np.argmax(Q[state])
    done = False
    while not done:
        path.append(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = np.argmax(Q[next_state])
        state, action = next_state, next_action
    return path


rewards_q_log, Q_q = train("q_learning")
rewards_sarsa_log, Q_s = train("sarsa")

window = 50
rewards_q = np.convolve(rewards_q_log, np.ones(window)/window, mode='valid')
rewards_sarsa = np.convolve(rewards_sarsa_log, np.ones(window)/window, mode='valid')

plt.plot(rewards_q, label="Q-learning")
plt.plot(rewards_sarsa, label="SARSA")
plt.xlabel("Эпизод")
plt.ylabel("Суммарное вознаграждение")
plt.legend()
plt.title("Сравнение Q-learning и SARSA")
#plt.ylim(-2000,0)
plt.show()

print(optimal(Q_q))
print("__")
print(optimal(Q_s))