# -*- coding: utf-8 -*


CARLA_HOST_ADDRESS = '192.168.1.98'

# CARLA image shape after resize
CARLA_IMG_HEIGHT = 60
CARLA_IMG_WIDTH = 80

# DQfD pre-training step number
PRE_TRAIN_STEP_NUM = 5000

EXPERIENCE_REPLAY_BUFFER_SIZE = 1000  # experience replay buffer size
DEMO_BUFFER_SIZE = 500 * 50
REPLAY_BUFFER_SIZE = DEMO_BUFFER_SIZE * 2

BATCH_SIZE = 64  # size of minibatch
ACTION_DIM = 2  # action/output dim of DQfD
STATE_DIM = 4  # state/input dim of DQfd
LEARNING_RATE = 0.001  # optimizer learning rate

GAMMA = 0.99  # discount factor for Q

