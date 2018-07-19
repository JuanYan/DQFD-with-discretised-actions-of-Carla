
#----------------------------the main function------------------------------
### It calls the demo recording module, the pre-train module and also the self paly module.

from Carla_demo_record import carla_demo
import config
from CustomEnv import CarlaEnv
from DQfD_model import Agent, Transition
from DQfD_Carla import dqfd_replay


exp = CarlaEnv(config.TARGET)
exp.reset()


demo_transitions = carla_demo(exp)
agent = Agent(demo_transitions)
agent.replay_memory_push(demo_transitions)
agent.demo_memory_push(demo_transitions)

#pretrainning
agent.pre_train()

# replay
dqfd_replay(exp, agent)
