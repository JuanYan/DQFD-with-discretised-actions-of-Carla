#  -*- coding: utf-8 -*-


import random
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Carla
# add carla to python path
if sys.platform == "linux":
    sys.path.append('/home/jy18/CARLA_0.8.3/PythonClient')

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings


# ----------------------------Parameters --------------------------------
CAPACITY = 5000
DEMO_SIZE = 2000
PLAY_SIZE = 3000
TARGET = np.array([158.08, 27.18])  # the target location point 134 on the map
FRAME_MAX = 1000  # if the agent has not arrived at the target within the given frames/time, demonstration fails.
BATCH_SIZE = 128


# ------------------------------Carla ------------------------------------

# The environment provides s, a, r(s), and transition s'
# a=[0,1] referring to moving left and right. s is represented by the current screen pixes subtracting the previous screen pixes.

def carla_init(client):
    """

    :param client:
    :return:
    """
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=20,
        NumberOfPedestrians=40,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel='Epic')

    # CAMERA
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

    # LIDAR
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


def carla_meas_pro(measurements):
    """

    :param measurements:
    :return:
    """
    pos_x = measurements.player_measurements.transform.location.x
    pos_y = measurements.player_measurements.transform.location.y
    speed = measurements.player_measurements.forward_speed * 3.6  # m/s -> km/h
    col_cars = measurements.player_measurements.collision_vehicles
    col_ped = measurements.player_measurements.collision_pedestrians
    col_other = measurements.player_measurements.collision_other
    other_lane = 100 * measurements.player_measurements.intersection_otherlane
    offroad = 100 * measurements.player_measurements.intersection_offroad
    agents_num = len(measurements.non_player_agents)

    meas = {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'speed': speed,
        'col_damage': col_cars + col_ped + col_other,
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
    dis = np.linalg.norm(pose - TARGET)

    if dis < 1:
        done = 1  # final state arrived!
    else:
        done = 0

    meas['dis'] = dis  # distance to target

    return meas, done


def cal_reward(meas_old, meas_new):
    """

    :param meas_old:
    :param meas_new:
    :return:
    """

    def delta(key):
        return meas_old[key] - meas_new[key]

    return 1000 * delta('dis') + 0.05 * delta('speed') - 0.00002 * delta('col_damage') \
           - 2 * delta('offroad') - 2 * delta('other_lane')


def carla_demo(client):
    """
    run carla and record the measurements, the images, and the control signals
    :param client:
    :return:
    """
    global memory

    episode_num = 1

    # file name format to save images
    out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'

    for episode in range(0, episode_num):
        # re-init client for each episode
        carla_init(client)
        # save all the measurement from frames
        measurement_list = []
        meas_old = None
        images_old = None

        for frame in range(0, FRAME_MAX):
            print('Running at Frame ', frame)

            # read new measurement
            measurements, images_new = client.read_data()

            # cal control signal
            control = measurements.player_measurements.autopilot_control
            control.steer += random.uniform(-0.1, 0.1)
            client.send_control(control)
            # client.send_control(
            #     steer=random.uniform(-1.0, 1.0),
            #     throttle=0.5,
            #     brake=0.0,
            #     hand_brake=False,
            #     reverse=False)

            # cal measurement
            meas_new, done = carla_meas_pro(measurements)

            # calculate and save reward into memory
            if meas_old:
                reward = cal_reward(meas_old, meas_new)
                memory.demopush(meas_old, images_old, control, reward, meas_new, images_new)

            # save image to disk
            for name, images in images_new.items():
                filename = out_filename_format.format(episode, name, frame)
                images.save_to_disk(filename)

            # save measurement
            measurement_list.append(meas_new)
            meas_old, images_old = meas_new, images_new

            # check for end condition
            if done:
                print('Target achieved!')
                break

        if not done:
            print("Target not achieved!")

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


Transition = namedtuple('Transition', 'meas_old, images_old, control, reward, meas_new, images_new')


# the replay memory
class ExperienceReplay(object):

    def __init__(self, capacity, demosize, playsize):
        self.capacity = capacity
        self.demosize = demosize
        self.playsize = playsize
        self.playmemory = []
        self.demomemory = []

    def demopush(self, *args):
        if len(self.demomemory) < DEMO_SIZE:
            self.demomemory.append(Transition(*args))
        else:
            return

    def playpush(self):
        pass

    def replaySample(self, batchsize):
        return random.sample(self.playmemory + self.demomemory, batchsize)


def loss():
    pass


def optimize_model(loss):
    pass


def select_action():
    pass


# Initialisation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ExperienceReplay(CAPACITY, DEMO_SIZE, PLAY_SIZE)

with make_carla_client('localhost', 2000) as client:
    print('Carla Client connected')

    # pre-trainning with only demonstration transitions
    # pretrain_iteration = 100
    # update_frequency = 20

    carla_demo(client)

    # # trainning with prioritized memory
    # for t in range(pretrain_iteration):
    #     pass
