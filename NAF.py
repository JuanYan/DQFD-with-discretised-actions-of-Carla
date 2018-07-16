# coding: utf-8

# In[1]:


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from gym import wrappers


# In[2]:


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


# In[3]:


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


# In[4]:


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (1 - 0)
        action += 0
        return action

    def _reverse_action(self, action):
        action -= 0
        action /= (1 - 0)
        action = action * 2 - 1
        return actions


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# In[5]:


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def hard_delayed_update(target, source, update_freq):
    global steps
    if steps % update_freq == 0 and steps > 0:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def calculate_eps(steps, start, end, decay):
    return end + (start - end) * math.exp(-1. * steps / decay)


class Policy(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        # self.bn0 = nn.BatchNorm1d(num_inputs)
        # self.bn0.weight.data.fill_(1)
        # self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn1.weight.data.fill_(1)
        # self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.bn2.weight.data.fill_(1)
        # self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        # self.V.weight.data.mul_(0.1)
        # self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs)
        # self.mu.weight.data.mul_(0.1)
        # self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)
        # self.L.weight.data.mul_(0.1)
        # self.L.bias.data.mul_(0.1)

        self.tril_mask = Variable(torch.tril(torch.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs))).unsqueeze(0))

    def forward(self, inputs):
        x, u = inputs
        # x = self.bn0(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        V = self.V(x)
        mu = F.tanh(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF:

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space,
                 optimizer, loss_func, lr, eps_s, eps_e, eps_d, update_freq):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = Policy(hidden_size, num_inputs, action_space)
        self.target_model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_func = loss_func

        self.gamma = gamma
        self.tau = tau
        self.eps_s = eps_s
        self.eps_e = eps_e
        self.eps_d = eps_d
        self.update_freq = update_freq

        hard_update(self.target_model, self.model)

    def select_action(self, state, exploration=None):
        self.model.eval()
        mu, _, _ = self.model((Variable(state, volatile=True), None))
        self.model.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        s = Variable(torch.cat(batch.state))
        a = Variable(torch.cat(batch.action))
        r = Variable(torch.cat(batch.reward))
        n = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

        _, Q, _ = self.model((s, a))
        V = Variable(torch.zeros(len(batch.state)).type(Tensor))
        _, _, V[mask] = self.target_model((n, None))
        V.volatile = False

        loss = self.loss_func(Q, (V * self.gamma) + r)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        # soft_update(self.target_model, self.model, self.tau)
        hard_delayed_update(self.target_model, self.model, self.update_freq)

        return loss.data[0]


# In[14]:


# env_name = 'simple_spread'
env_name = 'simple'

env = make_env(env_name)
env.seed(0)

# hyperparameters
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
LR = 0.001
TARGET_UPDATE_FREQ = 200
NUM_EPISODES = 200
MEM_SIZE = 10000
TAU = 0.001
HIDDEN_SIZE = 64
MAX_EPISODE_LENGTH = 30
num_inputs = env.observation_space[0].shape[0]
action_space = env.action_space[0].n
optimizer = optim.Adam
loss_func = F.mse_loss
noise_scale = 0.3
final_noise_scale = 0.1
exploration_end = 200

# model initialization
agents = []
for i in range(env.n):
    agents.append(NAF(GAMMA, TAU, HIDDEN_SIZE, num_inputs, action_space, optimizer, loss_func, LR,
                      EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE_FREQ))
    if use_cuda:
        agents.cuda()

# memory initilization
memory = ReplayMemory(MEM_SIZE)
ounoise = OUNoise(env.action_space[0].n)

steps = 0
loss_all = []
episode_durations = []

# In[25]:


MAX_EPISODE_LENGTH = 100

# In[29]:


for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    observations = np.stack(env.reset())
    observations = Tensor(observations)

    episode_reward = 0

    ounoise.scale = (noise_scale - final_noise_scale) * max(0, exploration_end -
                                                            i_episode) / exploration_end + final_noise_scale
    ounoise.reset()
    for i_step in range(MAX_EPISODE_LENGTH):

        actions = torch.stack([agent.select_action(obs, None) for agent, obs in zip(agents, observations)])

        next_observations, rewards, dones, _ = env.step(actions.numpy())
        next_observations = Tensor(next_observations)
        episode_reward += rewards[0]
        rewards = Tensor([rewards[0]])

        # if it is the last step we don't need next obs
        if i_step == MAX_EPISODE_LENGTH:
            next_observations = None

        # Store the transition in memory
        memory.push(observations, actions, next_observations, rewards)

        # Move to the next state
        observations = next_observations

        # use experience replay
        for i in range(env.n):
            if len(memory) > BATCH_SIZE:
                batch = Transition(*zip(*memory.sample(BATCH_SIZE)))
                loss = agents[i].update_parameters(batch)
                loss_all.append(loss)

        if i_step == MAX_EPISODE_LENGTH - 1:
            episode_durations.append(episode_reward)
            plot_durations()

        if i_episode % 10 == 0:
            env.render()

print('complete')
# env.close()


# In[30]:


np.mean(episode_durations[-100::])

# In[31]:


plt.plot(loss_all)
plt.ylim(0, 5)

# In[13]:


rewards

# In[ ]:


for i_episode in range(NUM_EPISODES):
    ounoise.scale = (noise_scale - final_noise_scale) * max(0, exploration_end -
                                                            i_episode) / exploration_end + final_noise_scale
    ounoise.reset()


