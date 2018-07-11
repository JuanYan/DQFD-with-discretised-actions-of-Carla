#  -*- coding: utf-8 -*-


import sys
import pickle
import itertools
import collections
import torch
import config
import pandas as pd
from DQfD_model import Agent, Transition
from memory import Memory, SumTree

# Carla
# add carla to python path
if sys.platform == "linux":
    sys.path.append('/home/jy18/CARLA_0.8.3/PythonClient')

from CustomEnv import CarlaEnv


def carla_demo(exp):
    """
    DQfD_CartPole carla and record the measurements, the images, and the control signals
    :param
    :return:
    """
    demomem = Memory(config.DEMO_BUFFER_SIZE)

    # file name format to save images
    out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'

    for episode in range(0, config.CARLA_DEMO_EPISODE):
        # re-init client for each episode
        exp.reset()
        # save all the measurement from frames
        measurements_list = []
        action_list = []
        reward_list = []

        for frame in range(0, config.CARLA_DEMO_FRAME):
            print('Running at Frame ', frame)

            if not exp.pre_measurements:
                action = None
            else:
                control = measurements.player_measurements.autopilot_control
                action_no, action = exp.action_discretize(control)
                actionprint = {
                    'action_number': action_no,
                    'steer': action.steer,
                    'throttle': action.throttle,
                    'brake': action.brake
                }
                action_list.append(actionprint)

            exp.cur_measurements, exp.cur_image, reward, done, measurements = exp.step(action)
            reward_list.append(reward)
            measurements_list.append(exp.cur_measurements)


            # calculate and save reward into memory
            # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'n_reward'))

            if exp.pre_measurements:
                pre_state = [exp.pre_measurements, exp.pre_image]
                cur_state = [exp.cur_measurements, exp.cur_image]
                transition = Transition(pre_state, action_no, cur_state, reward, torch.zeros(1))
                demomem.push(transition)

            # save image to disk
            for name, images in exp.cur_image.items():
                filename = out_filename_format.format(episode, name, frame)
                images.save_to_disk(filename)

            # Todo: remember to do the same in the self exploring part
            exp.pre_measurements, exp.pre_image = exp.cur_measurements, exp.cur_image

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

    return demomem

# #
# def dqfd_eval():
#     # create Carla env
#
#     # load demo transitions
#     demo_transitions = load_demo(config.DEMO_PICKLE_FILE)
#
#     # use the demo data to pre-train network
#     agent = Agent(demo_transitions = demo_transitions)
#     agent.pre_train()
#
#     # training loop
#     episode, replay_full_episode = 0, None
#     for i_episode in range(config.EPISODE_NUM):
#         done, n_reward, state = False, None, env.reset()
#         transitions = []
#         # transition_queue = collections.deque(maxlen=config.TRAJECTORY_NUM)
#         for step in itertools.count(10):
#             action = agent.e_greedy_select_action(state)
#             next_state, reward, done, info = env.step(action)
#             reward = torch.Tensor([reward])
#
#             # storing transition in a temporary replay buffer in order to calculate n-step returns
#             transitions.insert(0, Transition(state, action, next_state, reward, torch.zeros(1)))
#             state = next_state
#             gamma = 1
#             new_trans = []
#             for trans in transitions:
#                 new_trans.append(trans._replace(n_reward=trans.n_reward + gamma * reward))
#                 gamma = gamma * config.Q_GAMMA
#             transitions = new_trans
#
#             # if the episode isn't over, get the next q val and add the 10th transition to the replay buffer
#             # otherwise push all transitions to the buffer
#             if not done:
#                 q_val = agent.policy_net(torch.from_numpy(next_state)).data
#                 if len(transitions) >= 10:
#                     last_trans = transitions.pop()
#                     last_trans = last_trans._replace(n_reward=last_trans.n_reward + gamma * q_val.max(1)[0].cpu())
#                     agent.replay_memory.push(last_trans)
#                 state = next_state
#
#             else:
#                 for trans in transitions:
#                     agent.replay_memory.push(trans)
#
#             if agent.replay_memory.is_full:
#                 # TODO: check again
#                 agent.train()
#
#             if done:
#                 print("episode: %d, memory length: %d  epsilon: %f" % (i_episode, len(agent.replay_memory), agent.epsilon))
#                 break
#
#         # Update the target network
#         if i_episode % 100 == 0:
#             agent.update_target_net()
#     env.close()


if __name__ == "__main__":
    # dqfd_eval()

    exp = CarlaEnv(config.TARGET)
    demomem = carla_demo(exp)
    # agent = Agent(demo_transitions = demo_transitions)
    # agent.pre_train()






    # pre-trainning with only demonstration transitions
    # pretrain_iteration = 100
    # update_frequency = 20




    # ----------------------------pretrain --------------------------------



    # # trainning with prioritized memory
    # for t in range(pretrain_iteration):[]
    #     pass
