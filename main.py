
#----------------------------the main function------------------------------
### It calls the demo recording module, the pre-train module and also the self paly module , and also the test module.

from Carla_demo_record import carla_demo
import config
from CustomEnv import CarlaEnv
from DQfD_model import Agent, Transition
from Carla_replay import dqfd_replay
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd








exp = CarlaEnv(config.TARGET)
exp.reset()
plt.ion()

demo_transitions, episode_reward = carla_demo(exp)

with open(config.CARLA_DEMO_FILE, 'wb') as f:
    pickle.dump(demo_transitions, f)

agent = Agent(demo_transitions)
agent.replay_memory_push(demo_transitions)
agent.demo_memory_push(demo_transitions)

#pretrainning
agent.pre_train()

with open(config.CARLA_PRETRAIN_FILE, 'wb') as f:
    pickle.dump(agent, f)
    print("Pretrain parameters achevied!")

# replay
episode_replay_reward = dqfd_replay(exp, agent)


with open(config.CARLA_TRAIN_FILE, 'wb') as f:
    pickle.dump(agent, f)
    print("Trained parameters achevied!")

reward_df = pd.DataFrame(episode_replay_reward)
reward_df.to_csv('_episode_reward.csv')



plt.ioff()
plt.show()

plt.figure(2)
plt.plot(episode_replay_reward)
plt.ylabel('reward')
plt.xlabel('episode')
plt.show()

exp.close()

