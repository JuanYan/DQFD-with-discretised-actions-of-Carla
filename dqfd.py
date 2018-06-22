#  -*- coding: utf-8 -*-


import sys
sys.path.append('/home/jy18/CARLA_0.8.3/PythonClient')  # add carla to python path
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from PIL import Image
from itertools import count
from itertools import count


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


device = torch.device('cuda')
capacity = 5000
demosize = 2000
playsize = 3000
target = np.array([158.08, 27.18])# the target location point 134 on the map
frame_max = 1000  # if the agent hasnot arrived at the target within the given frames/time, demonstration fails.
batchsize = 128
Transition = namedtuple('Transition', 'meas_old, images_old, control, reward, meas_new, images_new')




# The enviroment provides s, a, r(s), and transition s'
# a=[0,1] referring to moving left and right. s is represented by the current screen pixes substracting the previous screen pixes.

def carla_init(client):
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=20,
        NumberOfPedestrians=40,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel='Epic')
    # settings.randomize_seeds()

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

    # define the starting point of the agent
    player_start = 140
    client.start_episode(player_start)
    print('Starting new episode at %r, %d...' % (scene.map_name, player_start))




def carla_observe(client):

    measurements, images = client.read_data()
    player_measurements = measurements.player_measurements
    pos_x = player_measurements.transform.location.x
    pos_y = player_measurements.transform.location.y
    speed = player_measurements.forward_speed * 3.6  # m/s -> km/h
    col_cars = player_measurements.collision_vehicles
    col_ped = player_measurements.collision_pedestrians
    col_other = player_measurements.collision_other
    other_lane = 100 * player_measurements.intersection_otherlane
    offroad = 100 * player_measurements.intersection_offroad
    agents_num = len(measurements.non_player_agents)

    meas= {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'speed': speed,
        'col_damage': col_cars+col_ped+col_other,
        'other_lane': other_lane,
        'offroad': offroad,
        'agents_num': agents_num,
    }

    message = 'Vehicle at ({:.1f}, {:.1f}), '
    message += '{:.0f} km/h, '
    message += 'Collision: {{vehicles={:.0f}, pedestrians={:.0f}, other={:.0f}}}, '
    message += '{:.0f}% other lane, {:.0f}% off-road, '
    message += '({:d} non-player agents in the scene)'
    message = message.format(pos_x, pos_y, speed, col_cars, col_ped, col_other, other_lane, offroad, agents_num)
    print(message)

    pose = np.array([pos_x, pos_y])
    dis = np.linalg.norm(pose - target)

    if dis < 1:
        done=1  #final state arrived!
    else:
        done = 0

    meas['dis']=dis   #distance to target

    return meas, images, done






# run carla and record the measurements, the images, and the control signals
def carla_demo(client):
    episode_num = 2

    out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'

    for episode in range(0, episode_num):
        carla_init(client)
        meas_old, images_old, done = carla_observe(client)
        measurement_list = []
        for frame in range(0, frame_max):

            measurements, sensor_data = client.read_data()
            control = measurements.player_measurements.autopilot_control
            control.steer += random.uniform(-0.1, 0.1)
            # client.send_control(control)
            client.send_control(
                steer=random.uniform(-1.0, 1.0),
                throttle=0.5,
                brake=0.0,
                hand_brake=False,
                reverse=False)



            meas_new, images_new, done = carla_observe(client)
            measurement_list.append(meas_new)

            reward = 1000*(meas_old['dis']-meas_new['dis'])+0.05*(meas_old['speed']-meas_new['speed'])-0.00002*(meas_old['col_damage']-meas_new['col_damage']) \
                     -2*(meas_old['offroad']-meas_new['offroad'])-2*(meas_old['other_lane']-meas_new['other_lane'])

            memory.demopush(meas_old, images_old, control, reward, meas_new, images_new)

            for name, images in sensor_data.items():
                filename = out_filename_format.format(episode, name, frame)
                images.save_to_disk(filename)



            meas_old = meas_new
            images_old = images_new

            if done:
                print('Target achieved!')
                break


        measurement_df = pd.DataFrame(measurement_list)
        measurement_df.to_csv('_measurements%d.csv' % episode)






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

class ExperienceReplay(object):

    def __init__(self, capacity, demosize, playsize):
        self.capacity = capacity
        self.demosize = demosize
        self.playsize = playsize
        self.playmemory = []
        self.demomemory = []

    def demopush(self, *args):
        if len(self.demomemory) < demosize:
            self.demomemory.append(Transition(*args))
        else:
            return


    def playpush(self):
        pass

    def replaySample(self, batchsize):
        return random.sample(self.memory, batch_size)


def loss():
    pass


def optimize_model(loss):
    pass


def select_action():
    pass


# Initialisation

memory = ExperienceReplay(capacity, demosize, playsize)


with make_carla_client('localhost', 2000) as client:
    print('CarlaClient connected')

    # pre-trainning with only demonstration transitions
    pretrain_iteration = 100
    update_frequency = 20

    carla_demo(client)


    # trainning with prioritized memory
    for t in range(pretrain_iteration):
        pass