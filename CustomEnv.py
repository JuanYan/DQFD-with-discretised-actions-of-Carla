# -*- coding: utf-8 -*-
"""
Wrap gym and Carla into common interface
"""
import random
import numpy as np
import pandas as pd
import math
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
        self.pre_image = None
        self.cur_image = None
        self.pre_measurements = None
        self.cur_measurements = None

    def step(self, action):
        """
        :param action:
        :return: next_state, reward, done, info
        """
        self.carla_client.send_control(action)
        measurements, self.cur_image = self.carla_client.read_data()
        self.cur_measurements = self.extract_measurements(measurements)
        # Todo: Checkup the reward function with the images
        reward, done = self.cal_reward()
        return self.cur_measurements, self.cur_image, reward, done, measurements

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
        # camera1 = Camera('CameraDepth', PostProcessing='Depth')
        # camera1.set_image_size(800, 600)
        # camera1.set_position(0.30, 0, 1.30)
        # settings.add_sensor(camera1)

        # LIDAR
        # lidar = Lidar('Lidar32')
        # lidar.set_position(0, 0, 2.50)
        # lidar.set_rotation(0, 0, 0)
        # lidar.set(
        #     Channels=32,
        #     Range=50,
        #     PointsPerSecond=100000,
        #     RotationFrequency=10,
        #     UpperFovLimit=10,
        #     LowerFovLimit=-30)
        # settings.add_sensor(lidar)

        scene = self.carla_client.load_settings(settings)
        self.pre_state = None
        self.cur_state = None

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

    def cal_reward(self):
        """
        :param
        :return: reward, done
        """

        if self.pre_measurements is None:
            return 0.0, False

        def delta(key):
            return self.pre_measurements[key] - self.cur_measurements[key]

        def reward_func():
            return 1000 * delta('distance') + 0.05 * delta('speed') - 0.00002 * delta('col_damage') \
                   - 2 * delta('offroad') - 2 * delta('other_lane')

        # check distance to target
        done = self.cur_measurements['distance'] < 1  # final state arrived or not

        return reward_func(), done


    def carla_demo(self):
        """
        DQfD_CartPole carla and record the measurements, the images, and the control signals
        :param
        :return:
        """
        global memory

        # file name format to save images
        out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'

        for episode in range(0, config.CARLA_DEMO_EPISODE):
            # re-init client for each episode
            self.reset()
            # save all the measurement from frames
            measurements_list = []
            action_list=[]
            reward_list=[]

            for frame in range(0, config.CARLA_DEMO_FRAME):
                print('Running at Frame ', frame)

                if not self.pre_measurements:
                    action=None

                else:
                    action = measurements.player_measurements.autopilot_control
                    # action.steer += random.uniform(-0.1, 0.1)

                    #discretise the action space
                    print('Before processing, steer=%.5f,throttle=%.2f,brake=%.2f' % (
                    action.steer, action.throttle, action.brake))
                    steer = int(10 * action.steer)
                    throttle = int(action.throttle / 0.5)  # action.throttle= 0, 0.5 or 1.0
                    brake = int(action.brake)  # action.brake=0 or 1.0
                    actionnumber = steer << 3 | throttle << 1 | brake #map the action combination into the a numerical value
                    action.steer, action.throttle, action.brake = steer / 10.0, throttle * 0.5, brake * 1.0
                    print('After processing, steer=%.5f,throttle=%.2f,brake=%.2f' % (
                    action.steer, action.throttle, action.brake))
                    actionprint={
                        'action_number':actionnumber,
                        'steer':action.steer,
                        'throttle': action.throttle,
                        'brake':action.brake
                    }
                    action_list.append(actionprint)

                self.cur_measurements, self.cur_image, reward, done, measurements = self.step(action)
                reward_list.append(reward)
                measurements_list.append(self.cur_measurements)

                # calculate and save reward into memory
                if self.pre_measurements:
                    #push demo memory
                    pass


                # save image to disk
                for name, images in self.cur_image.items():
                    filename = out_filename_format.format(episode, name, frame)
                    images.save_to_disk(filename)

                #Todo: remember to do the same in the self exploring part
                self.pre_measurements, self.pre_image = self.cur_measurements, self.cur_image

                # check for end condition
                if done:
                    print('Target achieved!')
                    break

            if not done:
                print("Target not achieved!")

            # save measurements, actions and rewards
            measurement_df = pd.DataFrame(measurements_list)
            measurement_df.to_csv('_measurements%d.csv' % episode)
            action_df = pd.DataFrame(action_list)
            action_df.to_csv('_actions%d.csv' % episode)
            reward_df = pd.DataFrame(reward_list)
            reward_df.to_csv('_reward%d.csv' % episode)



    def close(self):
        """

        :return:
        """
        self.carla_client.disconnect()

