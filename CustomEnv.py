# -*- coding: utf-8 -*-
"""
Wrap gym and Carla into common interface
"""
import gym
import config


class CartPoleEnv:
    def __init__(self):
        """
        A custom env represent an env
        """
        self.env = gym.make("CartPole-v1")
        self.reset()

    def step(self, action):
        """

        :param action:
        :return: next_state, reward, done, info
        """
        return self.env.step(action)

    def reset(self):
        self.env.reset()

    @property
    def action_dim(self):
        return self.env.action_space.n

    @property
    def state_dim(self):
        """

        :return:
        """
        return 4

    def close(self):
        self.env.close()


class CarlaEnv:
    def __init__(self):
        pass

    def step(self, action):
        """

        :param action:
        :return: next_state, reward, done, info
        """
        next_state, reward, done, info = None, None, 0, {}
        return next_state, reward, done, info

    def reset(self):
        """

        :return:
        """
        pass

    def close(self):
        """

        :return:
        """
        pass

    @property
    def action_dim(self):
        """

        :return:
        """
        return 1

    @property
    def state_dim(self):
        """

        :return:
        """
        return 4
