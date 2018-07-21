#  -*- coding: utf-8 -*-

import config
import pandas as pd
from CustomEnv import CarlaEnv
import pickle
import matplotlib.pyplot as plt
from Carla_replay import dqfd_replay

# Carla
# add carla to python path

if __name__ == "__main__":
    #

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    # demo_transitions = load_demo(config.CARLA_DEMO_FILE)
    with open(config.CARLA_TRAIN_FILE, 'rb') as f:
        print("loading trained parameters...")
        agent = pickle.load(f)
        num_prameters = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(num_prameters, "Trained Prameter loaded!")

    episode_replay_reward = dqfd_replay(exp, agent)


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

