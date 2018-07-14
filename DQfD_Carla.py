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
from utils import rgb_image_to_tensor
from CustomEnv import CarlaEnv


# Carla
# add carla to python path
if sys.platform == "linux":
    sys.path.append('/home/jy18/CARLA_0.8.3/PythonClient')




def carla_demo(exp):
    """
    DQfD_CartPole carla and record the measurements, the images, and the control signals
    :param
    :return:
    """
    demomem = Memory(config.DEMO_BUFFER_SIZE)
    demo_transitions=[]

    # file name format to save images
    out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'

    for episode in range(0, config.CARLA_DEMO_EPISODE):
        # re-init client for each episode
        exp.reset()
        # save all the measurement from frames
        measurements_list = []
        action_list = []
        reward_list = []
        meas = None
        state= None

        for frame in range(0, config.CARLA_DEMO_FRAME):
            print('Running at Frame ', frame)

            if not meas:
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

            next_meas, next_state, reward, done, measurements = exp.step(action)
            reward_list.append(reward)
            measurements_list.append(next_meas)


            # calculate and save reward into memory
            # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'n_reward'))

            if meas:

                transition = Transition(state,
                                        torch.tensor([[action_no]]),
                                        torch.tensor([[reward]]),
                                        next_state,
                                        torch.zeros(1))   #TODO: use both the measurement and the image later
                demo_transitions.append(transition)

            # save image to disk
            for name, images in exp.cur_image.items():
                filename = out_filename_format.format(episode, name, frame)
                images.save_to_disk(filename)

            # Todo: remember to do the same in the self exploring part
            meas, state = next_meas, next_state

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

    return  demo_transitions


if __name__ == "__main__":
    # dqfd_eval()

    exp = CarlaEnv(config.TARGET)
    demo_transitions = carla_demo(exp)
    agent = Agent(demo_transitions)
    agent.replay_memory_push(demo_transitions)
    agent.demo_memory_push(demo_transitions)
    agent.pre_train()

    for i_episode in range(config.EPISODE_NUM):
        exp.reset()
        state = None
        meas = None
        next_state = None
        next_meas = None


        # transition_queue = collections.deque(maxlen=config.TRAJECTORY_NUM)
        for steps in itertools.count(config.EXPERIENCE_REPLAY_FRAME):

            print("Replay frame: %d , length of replaymemory %d" % (steps, len(agent.replay_memory)))
            action_no = agent.e_greedy_select_action(state)
            action = exp.reverse_action(action_no)
            next_meas, next_state, reward, done, _ = exp.step(action)

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

            if done:
                print("episode: %d, memory length: %d  epsilon: %f" % (i_episode, len(agent.replay_memory), agent.epsilon))
                break

        if steps % 100 == 0:
            agent.update_target_net()

        # Update the target network

    exp.close()

