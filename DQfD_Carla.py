#  -*- coding: utf-8 -*-


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

if __name__ == "__main__":
    # dqfd_eval()

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    # demo_transitions = load_demo(config.CARLA_DEMO_FILE)
    with open(config.CARLA_PRETRAIN_FILE, 'rb') as f:
        agent = pickle.load(f)

    for i_episode in range(config.EPISODE_NUM):
        exp.reset()
        state = None
        meas = None
        next_state = None
        next_meas = None
        offroad_list = []
        otherlane_list=[]

        # transition_queue = collections.deque(maxlen=config.TRAJECTORY_NUM)
        for steps in itertools.count(config.DEMO_BUFFER_SIZE):

            print("Replay frame: %d , length of replaymemory %d" % (steps, len(agent.replay_memory)))
            action_no = agent.e_greedy_select_action(state)
            action = exp.reverse_action(action_no)
            next_meas, next_state, reward, done, _ = exp.step(action)
            next_state = utils.rgb_image_to_tensor(next_state['CameraRGB'])
            offroad_list.append(next_meas['offroad'])
            otherlane_list.append(next_meas['other_lane'])

            # reset the enviroment if the car stay offroad or other_lane for 5 consequent steps
            if len(offroad_list) > 10:
                ar = np.array(offroad_list[-5:]).astype('int64')
                tag1 = np.bitwise_and.reduce(ar)
                br= np.array(otherlane_list[-10:]).astype('int64')
                tag2 = np.bitwise_and.reduce(br)
                tag = tag1 | tag2
                if tag:
                    exp.reset()

            if meas:
                transition = Transition(state,
                                        torch.tensor([[action_no]]),
                                        torch.tensor([[reward]]),
                                        next_state,
                                        torch.zeros(1))  # TODO: use both the measurement and the image later
                agent.replay_memory_push([transition])

            state = next_state
            meas = next_meas


            if agent.replay_memory.is_full:
                # TODO: check again
                agent.train()
            #
            # if done:
            #     print("episode: %d, memory length: %d  epsilon: %f" % (i_episode, len(agent.replay_memory), agent.epsilon))
            #     break
            if steps % 100 == 0:
                agent.update_target_net()



        # Update the target network

    exp.close()

