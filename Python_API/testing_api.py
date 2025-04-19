from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Pytorch part is not Not functional yet
here as a skeleton for Ml-agents api
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_step = LiquidTimeStep(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid_step(x[:, t, :], h)
        output = self.output_layer(h)
        return output


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Actor:
    def __init__(
        self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        # epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:

            target = (
                reward
                + self.gamma
                * torch.max(
                    self.model(torch.tensor(next_state, dtype=torch.float32))
                ).item()
            )

            target_f = self.model(torch.tensor(state, dtype=torch.float32)).numpy()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(
                torch.tensor(target_f),
                self.model(torch.tensor(state, dtype=torch.float32)),
            )
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


def LLM_Main():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 2  # Output size for regression

    # Create the model
    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)

    # Define Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


def DQN_main():
    batch_size = 32
    num_episodes = 1000
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


def Unity_main():
    env = UnityEnvironment(file_name="../../../../../Maze gen/Builds/Maze_seeded.app")
    env.reset()
    behaviour_names = env.behavior_specs.keys()
    behaviour_name = list(behaviour_names)[0]
    spec = env.behavior_specs[behaviour_name]
    n_action = spec.action_spec.continuous_size

    # getting first observation to set params
    decision_steps, terminal_steps = env.get_steps(behaviour_name)
    n_agents = len(decision_steps.agent_id)
    if n_agents < 1:
        raise Exception("No agents")

    # give random action for first step
    actions = np.random.randn(n_agents, n_action)
    action_tuple = ActionTuple(continuous=actions)
    env.set_actions(behaviour_name, action_tuple)
    env.step()
    decision_steps, terminal_steps = env.get_steps(behaviour_name)
    # for __ in range(5):
    #     total_ind_rw = np.zeros(n_agents)
    #     for _ in range(200):
    #         actions = np.random.randn(n_agents, n_action)
    #         action_tuple = ActionTuple(continuous=actions)
    #         env.set_actions(behaviour_name, action_tuple)
    #         env.step()
    #         decision_steps, terminal_steps = env.get_steps(behaviour_name)
    #         total_ind_rw += decision_steps.reward
    #     print(total_ind_rw)
    #     print(np.sum(total_ind_rw))
    #     print()

    # exit()
    # one actor netwok decides action of n_agets in turn
    obs_in_use = np.zeros(n_agents, 8)
    prev_obs_in_use = obs_in_use.copy()
    actor = Actor(
        8,
        n_action,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        buffer_size=10000,
    )

    total_reward = 0
    for _ in range(2):
        obs = decision_steps.obs

        for k in range(n_agents):
            obs_in_use[k, :4] = obs[0][k][2:-1:3]
            obs_in_use[k, 4:] = obs[1][k]

            # individually epsilon-greedy
            action = actor.act(obs_in_use)
            actions[k] = action

        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behaviour_name, action_tuple)
        env.step()

        prev_obs_in_use[:] = obs_in_use[:]

        decision_steps, terminal_steps = env.get_steps(behaviour_name)
        obs = decision_steps.obs
        rewards = decision_steps.reward

        for k in range(n_agents):
            obs_in_use[k, :4] = obs[0][k][2:-1:3]
            obs_in_use[k, 4:] = obs[1][k]
            actor.remember(prev_obs_in_use[k], actions[k], rewards[k], obs_in_use[k])

            state = next_state
            total_reward += rewards[k]
            actor.replay(32)

    env.close()


if __name__ == "__main__":
    Unity_main()
