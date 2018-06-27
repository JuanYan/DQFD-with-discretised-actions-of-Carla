# -*- coding: utf-8 -*
"""
Deep Q-learning from Demonstrations
"""
import numpy as np
import random
import torch
from memory import Memory
from utils import lazy_property


class DQfD:
    def __init__(self,
                 replay_buffer_size,
                 demo_buffer_size,
                 demo_transitions=None):
        # replay_memory stores both demo data and generated data
        self.replay_memory = Memory(capacity=replay_buffer_size, permanent_size=len(demo_transitions))
        # demo_memory only store demo data
        self.demo_memory = Memory(capacity=demo_buffer_size, permanent_size=demo_buffer_size)
        # add demo data to both demo_memory & replay_memory
        self.replay_memory_push(demo_transitions)
        self.demo_memory_push(demo_transitions)

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
