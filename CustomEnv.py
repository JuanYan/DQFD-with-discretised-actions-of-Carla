# -*- coding: utf-8 -*-
"""
Wrap gym and Carla into common interface
"""
import random
import numpy as np
from carla.client import CarlaClient
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings

import config


class CarlaEnv:
    def __init__(self, target):
        self.carla_client = CarlaClient(
            config.CARLA_HOST_ADDRESS, config.CARLA_HOST_PORT)
        self.carla_client.connect()
        self.target = target
        self.pre_measurement = None
        self.cur_measurement = None

    def step(self, action):
        """
        :param action:
        :return: next_state, reward, done, info
        """
        self.carla_client.send_control(action)
        measurements, raw_sensor = self.carla_client.read_data()
        # Todo: Checkup the reward function with the images
        reward, done = self.cal_reward(measurements)
        return raw_sensor, reward, done, {}

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
        print('Starting new episode at %r, %d...' %
              (scene.map_name, player_start))

        # TODO: read and return status after reset
        return

    def extract_measurements(self, measurements):
        """
        extract custom measurement data from carla measurement
        :param measurements:
        :return: custom measurement data dict
        """
        p_meas = measurements.player_measurements
        pos_x = p_meas.transform.location.x
        pos_y = p_meas.transform.location.y
        speed = p_meas.forward_speed * 3.6  # m/s -> km/h
        col_cars = p_meas.collision_vehicles
        col_ped = p_meas.collision_pedestrians
        col_other = p_meas.collision_other
        other_lane = 100 * p_meas.intersection_otherlane
        offroad = 100 * p_meas.intersection_offroad
        agents_num = len(measurements.non_player_agents)
        distance = np.linalg.norm(np.array([pos_x, pos_y]) - self.target)
        meas = {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'speed': speed,
            'col_damage': col_cars + col_ped + col_other,
            'other_lane': other_lane,
            'offroad': offroad,
            'agents_num': agents_num,
            'distance': distance
        }
        message = 'Vehicle at %.1f, %.1f, ' % (pos_x, pos_y)
        message += '%.0f km/h, ' % (speed,)
        message += 'Collision: vehicles=%.0f, pedestrians=%.0f, other=%.0f, ' % (col_cars, col_ped, col_other,)
        message += '%.0f%% other lane, %.0f%% off-road, ' % (other_lane, offroad,)
        message += '%d non-player agents in the scene.' % (agents_num,)
        print(message)

        return meas

    def cal_reward(self, measurements):
        """

        :param measurements:
        :return: reward, done
        """
        assert measurements
        extracted_measurement = self.extract_measurements(measurements)
        self.pre_measurement, self.cur_measurement = self.cur_measurement, extracted_measurement

        # TODO: reward for the measurement, no previous measurement
        if self.pre_measurement is None:
            return 0, False

        def delta(key):
            return self.pre_measurement[key] - self.cur_measurement[key]

        def reward_func():
            return 1000 * delta('dis') + 0.05 * delta('speed') - 0.00002 * delta('col_damage') \
                   - 2 * delta('offroad') - 2 * delta('other_lane')

        # check distance to target
        done = extracted_measurement['distance'] < 1  # final state arrived or not

        return reward_func(), done

    def close(self):
        """

        :return:
        """
        self.carla_client.disconnect()

