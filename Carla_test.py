#  -*- coding: utf-8 -*-



import config
import pandas as pd
from CustomEnv import CarlaEnv
import utils
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
def dqfd_test(exp, agent):
    out_filename_format = '_imagetest/episode_{:0>4d}/{:s}/{:0>6d}'

    overall_reward = []

    for i_episode in range(config.TEST_EPISODE):
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
        for steps in range(config. REPLAY_FRAME):
            print("Test episode: %d, frame: %d , length of replaymemory %d" % (i_episode, steps, len(agent.replay_memory)))
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
                    break
                    print("Reset because of off road or intersection!")

            for name, images in exp.cur_image.items():
                filename = out_filename_format.format(i_episode, name, steps)
                images.save_to_disk(filename)

            state = next_state
            meas = next_meas

        overall_reward.append(episode_reward / steps)
        utils.plot_reward(overall_reward)
        print("Episode finished!")


    print("Test finished!")

    return  overall_reward



# Carla
# add carla to python path

if __name__ == "__main__":
    #

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    # demo_transitions = load_demo(config.CARLA_DEMO_FILE)
    with open(config.CARLA_TRAIN_FILE, 'rb') as f:
        print("loading train parameters...")
        agent = pickle.load(f)
        num_prameters = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(num_prameters, "Trained Prameter loaded!")

    episode_test_reward = dqfd_test(exp, agent)


    reward_df = pd.DataFrame(episode_test_reward)
    reward_df.to_csv('_episode_test_reward.csv')

    plt.plot(episode_test_reward)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()

    exp.close()

