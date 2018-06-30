# -*- coding: utf-8 -*
"""
Common configuration parameters
"""
USE_CUDA = False # to use GPU or not, False will disable using GPU even if available
# ----------------------------- Carla ----------------------------------------------
# CARLA_HOST_ADDRESS = "192.168.1.98" # Carla host address
# Carla on local machine, use 'localhost'
CARLA_HOST_ADDRESS = "localhost"
CARLA_HOST_PORT = 2000 # Port number

# CARLA image shape after resize
CARLA_IMG_HEIGHT = 60
CARLA_IMG_WIDTH = 80

# ----------------------------- DQfD ----------------------------------------------
PRE_TRAIN_STEP_NUM = 500  # DQfD pre-training step number

EXPERIENCE_REPLAY_BUFFER_SIZE = 1000  # experience replay buffer size
DEMO_BUFFER_SIZE = 500 * 50
REPLAY_BUFFER_SIZE = DEMO_BUFFER_SIZE * 2

BATCH_SIZE = 64  # size of mini batch
ACTION_DIM = 2  # action/output dim of DQfD
STATE_DIM = 4  # state/input dim of DQfd
LEARNING_RATE = 0.001  # optimizer learning rate

INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY = 200

Q_GAMMA = 0.99  # discount factor for Q
EPISODE_NUM = 3000
TRAJECTORY_NUM = 10  # n-step number for n-step TD-loss in both demo data and generated data

# loss related
SU_LOSS_MARGIN = 1  # supervised loss margin
SU_LOSS_LAMBDA = 1
N_STEP_LOSS_LAMBDA = 1

TARGET_UPDATE = 10
# --------------------- Files ----------------------------------------------------
DEMO_PICKLE_FILE = "./new_demo.p"
