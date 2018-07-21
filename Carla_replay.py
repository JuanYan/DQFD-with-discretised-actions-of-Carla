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
import matplotlib.pyplot as plt

plt.ion()

def dqfd_replay(exp, agent):

    overall_reward = []

    for i_episode in range(config.REPLAY_EPISODE):
        print("Environment reset...")
        exp.reset()
        state = None
        meas = None
        next_state = None
        next_meas = None
        offroad_list = []
        otherlane_list = []
        episode_reward = 0

        # transition_queue = collections.deque(maxlen=config.TRAJECTORY_NUM)
        for steps in itertools.count(config.DEMO_BUFFER_SIZE):
            frame_no = steps - config.DEMO_BUFFER_SIZE
            print("Replay episode: %d, frame: %d , length of replaymemory %d" % (i_episode, frame_no , len(agent.replay_memory)))
            action_no = agent.e_greedy_select_action(state)
            action = exp.reverse_action(action_no)
            next_meas, next_state, reward, done, _ = exp.step(action)
            next_state = utils.rgb_image_to_tensor(next_state['CameraRGB'])
            offroad_list.append(next_meas['offroad'])
            otherlane_list.append(next_meas['other_lane'])
            episode_reward += reward

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
                transition = Transition(meas,
                                        state,
                                        torch.tensor([[action_no]]),
                                        torch.tensor([[reward]]),
                                        next_state,
                                        next_meas,
                                        torch.zeros(1))  # TODO: use both the measurement and the image later
                agent.replay_memory_push([transition])

            state = next_state
            meas = next_meas


            if agent.replay_memory.is_full:
                print("Trainning!")
                agent.train()
            #
            # if done:
            #     print("episode: %d, memory length: %d  epsilon: %f" % (i_episode, len(agent.replay_memory), agent.epsilon))
            #     break
            if steps % 100 == 0:
                agent.update_target_net()

            if frame_no >= config. REPLAY_FRAME:
                overall_reward.append(episode_reward / config. REPLAY_FRAME)
                utils.plot_reward(overall_reward)
                print("Episode finished!")
                break

 #save the result every 20 episode
        if i_episode % 20 == 0:
            print("Saving prameters for the last 20 episodes")
            reward_df = pd.DataFrame(overall_reward)
            reward_df.to_csv('_episode_reward%d.csv' % i_episode)
            with open(config.CARLA_TRAIN_FILE, 'wb') as f:
                pickle.dump(agent, f)
                print("Trained parameters achevied!")

    print("Replay finished!")

    return  overall_reward


# Carla
# add carla to python path

if __name__ == "__main__":
    #

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    # demo_transitions = load_demo(config.CARLA_DEMO_FILE)
    with open(config.CARLA_PRETRAIN_FILE, 'rb') as f:
        print("loading pretrain parameters...")
        agent = pickle.load(f)
        num_prameters = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(num_prameters, "Prameter loaded!")

    episode_replay_reward = dqfd_replay(exp, agent)

    plt.ioff()
    plt.show()

    reward_df = pd.DataFrame(episode_replay_reward)
    reward_df.to_csv('_episode_reward.csv')

    with open(config.CARLA_TRAIN_FILE, 'wb') as f:
        pickle.dump(agent, f)
        print("Trained parameters achevied!")


    plt.plot(episode_replay_reward)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()

    exp.close()

