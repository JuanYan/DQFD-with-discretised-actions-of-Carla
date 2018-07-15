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
    # def __init__(self):
    #     super(DQN, self).__init__()
    #     self.l1 = nn.Linear(config.STATE_DIM, 24)
    #     self.l2 = nn.Linear(24, 24)
    #     self.l3 = nn.Linear(24, config.ACTION_DIM)
    #
    # def forward(self, x):
    #     x = F.relu(self.l1(x))
    #     x = F.relu(self.l2(x))
    #     x = F.relu(self.l3(x))
    #     return x.view(x.size(0), -1)


    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(28*32, 4*201)  # there is 84 output options for the action space

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'n_reward'))


class Agent:
    def __init__(self, demo_transitions=None):
        replay_buffer_size = config.REPLAY_BUFFER_SIZE
        demo_buffer_size = config.DEMO_BUFFER_SIZE
        # replay_memory stores both demo data and generated data
        self.replay_memory = Memory(capacity=replay_buffer_size, permanent_size=len(demo_transitions))
        # demo_memory only store demo data
        self.demo_memory = Memory(capacity=demo_buffer_size, permanent_size=demo_buffer_size)
        self.epsilon = config.INITIAL_EPSILON
        self.steps_done = 0
        #
        self.target_net = DQN().to(device, dtype=torch.double)
        self.policy_net = DQN().to(device,dtype=torch.double)

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
        if random.random() <= self.epsilon or state is None:
            return random.randint(0, config.ACTION_DIM - 1)
        else:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(device, dtype=torch.double)
            return self.policy_net(state.to(device, dtype=torch.double)).max(1)[1].view(1, 1).item()  # TODO:

    def pre_train(self):
        """
        pre train
        :return:
        """
        k = config.PRE_TRAIN_STEP_NUM
        print("Pre training for %d steps." % k)
        # for i in tqdm(range(k)):
        for i in range(k):
            self.train(pre_train=True)
            print('steps: %d' % i)
            if i % config.TARGET_UPDATE == 0:
                self.update_target_net()
                print('Target network updated!')
        print("Pre training done for %d steps." % k)

    def train(self, pre_train=False):
        """
        train Q network
        :param pre_train: if used for pre train or not
        :return:
        """

        # choose which memory to use
        mem = self.demo_memory if pre_train else self.replay_memory
        #  sample
        batch_id, batch_data, batch_weight = mem.sample(config.BATCH_SIZE)

        # extract data from each column
        batch = Transition(*zip(*batch_data.tolist()))  #array to list to transform

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)# TODO: change to target state when appropreiate
        non_final_next_states = torch.cat([torch.Tensor(s.double()) for s in batch.next_state if s is not None]).double()
        state_batch = torch.cat(batch.state).double()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).double()
        n_reward_batch = torch.cat(batch.n_reward).double()

        # # Compute Q(s_t, a) - the model computes Q(s_t), the action to take for the next state
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # calculate Q(s_t, a, \theta) under the current actions
        next_state_values = torch.zeros(config.BATCH_SIZE, device=device)         # Compute V(s_{t+1}) for all
        # next_state_values[non_final_mask] = self.policy_net(non_final_next_states).data.max(1)[0]  #next maximum state values  #DQN
        action_batch_next_state = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)  #DDQN
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_batch_next_state).squeeze().detach()  #DDQN
        expected_state_action_values = (next_state_values * config.Q_GAMMA) + reward_batch.squeeze(1)

        # calculating the q loss and n-step return loss
        q_loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), size_average=False)
        n_step_loss = F.mse_loss(state_action_values, n_reward_batch.unsqueeze(1), size_average=False)
        n_step_loss = 0


        # calculating the supervised loss
        if pre_train:
            action_dim = config.ACTION_DIM
            margins = (torch.ones(action_dim, action_dim) - torch.eye(action_dim)) * config.SU_LOSS_MARGIN
            batch_margins = margins[action_batch.data.squeeze().cpu()]
            state_action_values_with_margin = self.policy_net(state_batch) + batch_margins
            supervised_loss = (state_action_values_with_margin.max(1)[0].unsqueeze(1) - state_action_values).pow(2).sum()
        else:
            supervised_loss = 0.0


        loss = q_loss + config.SU_LOSS_LAMBDA * supervised_loss + config.N_STEP_LOSS_LAMBDA * n_step_loss

        # optimization step and logging
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        # self.optimizer.step()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
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
