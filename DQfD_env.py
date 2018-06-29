# -*- coding: utf-8 -*

import pickle
import itertools
import collections
import torch
import config
from DQfD_model import Agent, Transition
from CustomEnv import GymEnv


def load_demo(demo_file):
    """
    load demo transitions from file
    :param demo_file:
    :return:
    """
    with open(demo_file, 'rb') as f:
        # load demo transitions from pickle
        demos = pickle.load(f, encoding='latin1')
        # use only the first DEMO_BUFFER_SIZE transitions as demo
        demos = collections.deque(itertools.islice(demos, 0, config.DEMO_BUFFER_SIZE))
        assert len(demos) == config.DEMO_BUFFER_SIZE
        return demos


def dqfd_eval():
    # create gym env
    env = GymEnv()
    # load demo transitions
    demo_transitions = load_demo(config.DEMO_PICKLE_FILE)

    # use the demo data to pre-train network
    agent = Agent(demo_transitions=demo_transitions)
    agent.pre_train()

    # training loop
    episode, replay_full_episode = 0, None
    for i_episode in range(config.EPISODE_NUM):
        done, n_reward, state = False, None, env.reset()
        transitions = []
        # transition_queue = collections.deque(maxlen=config.TRAJECTORY_NUM)
        for step in itertools.count(10):
            action = agent.e_greedy_select_action(state)
            next_state, reward, done, info = env.step(action)
            reward = torch.Tensor([reward])

            # storing transition in a temporary replay buffer in order to calculate n-step returns
            transitions.insert(0, Transition(state, action, next_state, reward, torch.zeros(1)))
            state = next_state
            gamma = 1
            new_trans = []
            for trans in transitions:
                new_trans.append(trans._replace(n_reward=trans.n_reward + gamma * reward))
                gamma = gamma * config.Q_GAMMA
            transitions = new_trans

            # if the episode isn't over, get the next q val and add the 10th transition to the replay buffer
            # otherwise push all transitions to the buffer
            if not done:
                q_val = agent.policy_net(torch.from_numpy(next_state)).data
                if len(transitions) >= 10:
                    last_trans = transitions.pop()
                    last_trans = last_trans._replace(n_reward=last_trans.n_reward + gamma * q_val.max(1)[0].cpu())
                    agent.replay_memory.push(last_trans)
                state = next_state

            else:
                for trans in transitions:
                    agent.replay_memory.push(trans)

            if agent.replay_memory.is_full:
                # TODO: check again
                agent.train()

            if done:
                print("episode: %d, memory length: %d  epsilon: %f" % (i_episode, len(agent.replay_memory), agent.epsilon))
                break

        # Update the target network
        if i_episode % 100 == 0:
            agent.update_target_net()
    env.close()


if __name__ == "__main__":
    dqfd_eval()
