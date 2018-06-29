# -*- coding: utf-8 -*-
"""
Wrap gym and Carla into common interface
"""
import gym
import random
import numpy as np
from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings

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
        self.env.render()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    @property
    def action_dim(self):
        return self.env.action_space.n

    @property
    def state_dim(self):
        """

        :return:
        """
        return self.env.observation_space.shape[0]

    def close(self):
        self.env.close()


class CarlaEnv:
    def __init__(self, target):
        self.carla_client = make_carla_client(config.CARLA_HOST_ADDRESS, 2000)
        self.target = target

    def step(self, action):
        """

        :param action:
        :return: next_state, reward, done, info
        """
        self.carla_client.send_control(action)
        measurements, images_new = self.carla_client.read_data()
        meas_new, done = self._carla_meas_pro(measurements)

        next_state = images_new
        reward = 0  # TODO:
        # reward = self._cal_reward(meas_old, meas_new)
        return next_state, reward, done, {}

    def reset(self):
        """

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

        scene = self.carla_client.load_settings(settings)

        # define the starting point of the agent
        player_start = 140
        self.carla_client.start_episode(player_start)
        print('Starting new episode at %r, %d...' % (scene.map_name, player_start))

        # TODO: read and return status after reset
        return

    def _carla_meas_pro(self, measurements):
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
        dis = np.linalg.norm(pose - self.target)

        if dis < 1:
            done = 1  # final state arrived!
        else:
            done = 0

        meas['dis'] = dis  # distance to target

        return meas, done

    def _cal_reward(self, meas_old, meas_new):
        """

        :param meas_old:
        :param meas_new:
        :return:
        """

        def delta(key):
            return meas_old[key] - meas_new[key]

        return 1000 * delta('dis') + 0.05 * delta('speed') - 0.00002 * delta('col_damage') \
               - 2 * delta('offroad') - 2 * delta('other_lane')

    def close(self):
        """

        :return:
        """
        self.carla_client.close()

    @property
    def action_dim(self):
        """

        :return:
        """
        return 0

    @property
    def state_dim(self):
        """

        :return:
        """
        return 0
