# -*- coding: utf-8 -*
"""
Deep Q-learning from Demonstrations
"""
import numpy as np
import random
from collections import namedtuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import Memory
import config

torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
dtype = torch.cuda.DoubleTensor if config.USE_CUDA and torch.cuda.is_available() else torch.DoubleTensor


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(config.STATE_DIM, 24)
        self.l2 = nn.Linear(24, 24)
        self.l3 = nn.Linear(24, config.ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x.view(x.size(0), -1)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'n_reward'))


class Agent:
    def __init__(self,
                 demo_transitions=None):
        replay_buffer_size = config.REPLAY_BUFFER_SIZE
        demo_buffer_size = config.DEMO_BUFFER_SIZE
        # replay_memory stores both demo data and generated data
        self.replay_memory = Memory(capacity=replay_buffer_size, permanent_size=len(demo_transitions))
        # demo_memory only store demo data
        self.demo_memory = Memory(capacity=demo_buffer_size, permanent_size=demo_buffer_size)
        # add demo data to both demo_memory & replay_memory
        self.replay_memory_push(demo_transitions)
        self.demo_memory_push(demo_transitions)

        #
        self.epsilon = config.INITIAL_EPSILON
        self.steps_done = 0
        #
        self.target_net = DQN().to(device)
        self.policy_net = DQN().to(device)

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

    def e_greedy_select_action(self, state):
        """

        :param state:
        :return:
        """
        self.epsilon = config.FINAL_EPSILON + (config.INITIAL_EPSILON - config.FINAL_EPSILON) * \
                       np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1
        if random.random() <= self.epsilon:
            return random.randint(0, config.ACTION_DIM - 1)
        else:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(device)
            return self.policy_net(state).max(0)[1].view(1, 1).item()  # TODO:

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
        if not pre_train and not self.replay_memory.is_full:
            return  # for normal training, sample only after replay mem is full
        # choose which memory to use
        mem = self.demo_memory if pre_train else self.replay_memory
        # random sample
        batch_id, batch_data, batch_weight = mem.sample(config.BATCH_SIZE)
        # np.random.shuffle(batch_data)
        # extract data from each column
        batch = Transition(*zip(*batch_data))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.next_state if s is not None]).view(config.BATCH_SIZE, -1)
        state_batch = torch.cat(batch.state).view(config.BATCH_SIZE, -1)
        action_batch = torch.cat(batch.action).view(config.BATCH_SIZE, -1)
        reward_batch = torch.cat(batch.reward).view(config.BATCH_SIZE).double()
        n_reward_batch = torch.cat(batch.n_reward).view(config.BATCH_SIZE).double()

        # # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        #     # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all
        next_state_values = torch.zeros(config.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).data.max(1)[0]
        expected_state_action_values = (next_state_values * config.Q_GAMMA) + reward_batch

        # calculating the q loss and n-step return loss
        q_loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), size_average=False)
        n_step_loss = F.mse_loss(state_action_values, n_reward_batch.unsqueeze(1), size_average=False)

        # calculating the supervised loss
        action_dim = config.ACTION_DIM
        margins = (torch.ones(action_dim, action_dim) - torch.eye(action_dim)) * config.SU_LOSS_MARGIN
        batch_margins = margins[action_batch.data.squeeze().cpu()]
        state_action_values_with_margin = state_action_values + Variable(batch_margins).type(dtype)
        supervised_loss = (state_action_values_with_margin.max(1)[0].unsqueeze(1) - state_action_values_with_margin).pow(2)[:config.DEMO_BUFFER_SIZE].sum()

        loss = q_loss + config.SU_LOSS_LAMBDA * supervised_loss + config.N_STEP_LOSS_LAMBDA * n_step_loss

        # optimization step and logging
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        with torch.no_grad():
            abs_errors = torch.sum(torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)), dim=1)
            abs_errors = abs_errors.detach().numpy()

        self.replay_memory.batch_update(batch_id, abs_errors)  # update priorities for data in memory

    def update_target_net(self):
        """

        :return:
        """
        # Update the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
