# -*- coding: utf-8 -*
"""
Common configuration parameters
"""

# ----------------------------- Carla ----------------------------------------------
CARLA_HOST_ADDRESS = '192.168.1.98'

# CARLA image shape after resize
CARLA_IMG_HEIGHT = 60
CARLA_IMG_WIDTH = 80

# ----------------------------- DQfD ----------------------------------------------
PRE_TRAIN_STEP_NUM = 5000  # DQfD pre-training step number

EXPERIENCE_REPLAY_BUFFER_SIZE = 1000  # experience replay buffer size
DEMO_BUFFER_SIZE = 500 * 50
REPLAY_BUFFER_SIZE = DEMO_BUFFER_SIZE * 2

BATCH_SIZE = 64  # size of mini batch
ACTION_DIM = 2  # action/output dim of DQfD
STATE_DIM = 4  # state/input dim of DQfd
LEARNING_RATE = 0.001  # optimizer learning rate

INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
EPSILIN_DECAY = 0.999

Q_GAMMA = 0.99  # discount factor for Q
EPISODE_NUM = 300
TRAJECTORY_NUM = 10  # n-step number for n-step TD-loss in both demo data and generated data
# --------------------- Files ----------------------------------------------------
DEMO_PICKLE_FILE = "./demo.p"
