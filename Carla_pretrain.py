import itertools
import torch
import config
import pandas as pd
from DQfD_model import Agent, Transition
from CustomEnv import CarlaEnv
import utils
import pickle
import collections
import numpy as np


# Carla
# add carla to python path


def load_demo(demo_file):
    """
    load demo transitions from file
    :param demo_file:
    :return:
    """
    with open(demo_file, 'rb') as f:
        # load demo transitions from pickle
        demos = pickle.load(f)
        # use only the first DEMO_BUFFER_SIZE transitions as demo
        demos = collections.deque(itertools.islice(demos, 0, config.DEMO_BUFFER_SIZE))
        assert len(demos) == config.DEMO_BUFFER_SIZE
        return demos



if __name__ == "__main__":


    demo_transitions = load_demo(config.CARLA_DEMO_FILE)
    agent = Agent(demo_transitions)
    agent.replay_memory_push(demo_transitions)
    agent.demo_memory_push(demo_transitions)
    agent.pre_train()

    with open(config.CARLA_PRETRAIN_FILE, 'wb') as f:
        pickle.dump(agent, f)
        print("Parameters achevied!")