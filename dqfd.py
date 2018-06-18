#  -*- coding: utf-8 -*-



import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line



# The enviroment provides s, a, r(s), and transition s'
# a=[0,1] referring to moving left and right. s is represented by the current screen pixes substracting the previous screen pixes.

def Carla_init(client):

    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=20,
        NumberOfPedestrians=40,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()

    camera0 = Camera('CameraRGB')
    # Set image resolution in pixels.
    camera0.set_image_size(800, 600)
    # Set its position relative to the car in meters.
    camera0.set_position(0.30, 0, 1.30)
    settings.add_sensor(camera0)

    # Let's add another camera producing ground-truth depth.
    camera1 = Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(800, 600)
    camera1.set_position(0.30, 0, 1.30)
    settings.add_sensor(camera1)

    lidar = Lidar('Lidar32')
    lidar.set_position(0, 0, 2.50)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=32,
        Range=50,
        PointsPerSecond=100000,
        RotationFrequency=10,
        UpperFovLimit=10,
        LowerFovLimit=-30)
    settings.add_sensor(lidar)

    scene = client.load_settings(settings)

    #define the starting point of the agent
    player_start = 0
    client.start_episode(player_start)
    print('Starting new episode at %r, %d...' % scene.map_name, player_start)



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

with make_carla_client('localhost', 2000) as client:
    print('CarlaClient connected')

#pre-trainning with only demonstration transitions
pretrain_iteration = 100
update_frequency = 20
for t in range(pretrain_iteration):
    Carla_init(client)
    measurements, sensor_data = client.read_data()
    for name, measurement in sensor_data.items():
        filename = args.out_filename_format.format(episode, name, frame)
        measurement.save_to_disk(filename)

    pass


#trainning with prioritized memory
for t in count():
    Carla_init(client)
    measurements, sensor_data = client.read_data()
    pass
