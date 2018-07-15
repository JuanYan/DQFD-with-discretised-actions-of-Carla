#  -*- coding: utf-8 -*-


import sys
import itertools
import torch
import config
import pandas as pd
from DQfD_model import Agent, Transition
from memory import Memory
from CustomEnv import CarlaEnv
import utils
import pickle
import collections




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
    # dqfd_eval()

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    demo_transitions = load_demo(config.CARLA_DEMO_FILE)
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
            next_state = utils.rgb_image_to_tensor(next_state['CameraRGB'])


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

