# -*- coding: utf-8 -*
"""
Common configuration parameters
"""

import numpy as np


USE_CUDA = False # to use GPU or not, False will disable using GPU even if available
# ----------------------------- Carla ----------------------------------------------
# CARLA_HOST_ADDRESS = "192.168.1.98" # Carla host address
# Carla on local machine, use 'localhost'
CARLA_HOST_ADDRESS = "localhost"
CARLA_HOST_PORT = 2000 # Port number

# CARLA image shape after resize
CARLA_IMG_HEIGHT = 60
CARLA_IMG_WIDTH = 80

#CARLA demo
CARLA_DEMO_EPISODE = 1
CARLA_DEMO_FRAME = 40

TARGET = np.array([158.08, 27.18])  # the target location point 134 on the map



# ----------------------------- DQfD ----------------------------------------------
PRE_TRAIN_STEP_NUM = 500  # DQfD pre-training step number

EXPERIENCE_REPLAY_BUFFER_SIZE = 1000  # experience replay buffer size
DEMO_BUFFER_SIZE = 39    #500*50
REPLAY_BUFFER_SIZE = DEMO_BUFFER_SIZE * 2

BATCH_SIZE = 4  # size of mini batch
ACTION_DIM = 21*4  # action/output dim of DQfD
STATE_DIM = 4  # state/input dim of DQfd
LEARNING_RATE = 0.001  # optimizer learning rate

INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY = 200

Q_GAMMA = 0.99  # discount factor for Q
EPISODE_NUM = 1
TRAJECTORY_NUM = 10  # n-step number for n-step TD-loss in both demo data and generated data

# loss related
SU_LOSS_MARGIN = 1  # supervised loss margin
SU_LOSS_LAMBDA = 1
N_STEP_LOSS_LAMBDA = 1

TARGET_UPDATE = 100
# --------------------- Files ----------------------------------------------------
DEMO_PICKLE_FILE = "./new_demo.p"
