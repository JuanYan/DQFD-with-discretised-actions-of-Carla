#  -*- coding: utf-8 -*-



import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T





# The enviroment provides s, a, r(s), and transition s'
# a=[0,1] referring to moving left and right. s is represented by the current screen pixes substracting the previous screen pixes.


# Use pytorch to define the deep CNN for target network and policy network
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# the replay memory

class ExperienceReplay()

    def _init_(self, capacity, demosize, playsize):
        self.capacity = capacity
        self.demosize = demosize
        self.playsize = playsize
        self.playmemory = []
        self.demomemory = []

    def demoRecord(self, demosize):
        pass

    def playRecord(self):
        pass

    def replaySample(self, batchsize):
        pass




def loss():
    pass

def optimize_model(loss):
    pass

def select_action():
    pass



#Initialisation
a = ExperienceReplay(50000,20000,30000)
batchsize = 128
a.demoRecord()

#pre-trainning with only demonstration transitions
pretrain_iteration = 100
update_frequency = 20
for t in range(pretrain_iteration):
    pass


#trainning with prioritized memory
for t in count():
    pass
