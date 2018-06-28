# -*- coding: utf-8 -*
"""
Deep Q-learning from Demonstrations
"""
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from memory import Memory
import config


class DQfD(nn.Module):
    def __init__(self):
        super(DQfD, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, config.ACTION_DIM)  # TODO: linear and output dim

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent:
    def __init__(self,
                 replay_buffer_size,
                 demo_buffer_size,
                 demo_transitions=None):
        # replay_memory stores both demo data and generated data
        self.replay_memory = Memory(capacity=replay_buffer_size, permanent_size=len(demo_transitions))
        # demo_memory only store demo data
        self.demo_memory = Memory(capacity=demo_buffer_size, permanent_size=demo_buffer_size)
        # add demo data to both demo_memory & replay_memory
        self.replay_memory_push(demo_transitions)
        self.demo_memory_push(demo_transitions)

        #
        self.target_net = DQfD()
        self.policy_net = DQfD()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)

    def replay_memory_push(self, transitions):
        """
        Add transitions to replay_memory
        :param transitions: List of transitions
        :return:
        """
        for t in transitions:
            self.replay_memory.push(np.array(t, dtype=object))

    def demo_memory_push(self, transitions):
        """
        Add transitions to demo_memory
        :param transitions: List of transitions
        :return:
        """
        for t in transitions:
            self.demo_memory.push(np.array(t, dtype=object))

    def loss(self):
        raise NotImplemented()
        # TODO: loss functions

    def e_greedy(self):
        raise NotImplemented()

    def pre_train(self):
        """
        pre train
        :return:
        """
        k = config.PRE_TRAIN_STEP_NUM
        print("Pre training for %d steps." % k)
        for i in tqdm(range(k)):
            self.train(pre_train=True)
        print("Pre training done for %d steps." % k)

    def train(self, pre_train=True):
        """
        train Q network
        :param pre_train: if used for pre train or not
        :return:
        """
        if not pre_train and not self.replay_memory.full():
            return  # for normal training, sample only after replay mem is full
        # choose which memory to use
        mem = self.demo_memory if pre_train else self.replay_memory
        # random sample
        batch_id, batch_data, batch_weight = mem.sample(config.BATCH_SIZE)
        np.random.shuffle(batch_data)
        # extract data from each column
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, demo_data, n_step_reward_batch, \
        n_step_state_batch, n_step_done_batch, actual_n = list(zip(*batch_data))

        # TODO: eval policy net

        for i in tqdm(range(config.BATCH_SIZE)):
            self.target_net(state_batch[i].reshape([-1, config.STATE_DIM])).max(1)[0].detach()  # TODO: target net

        # optimize
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
