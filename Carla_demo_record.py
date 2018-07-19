

import torch
import config
import pandas as pd
from DQfD_model import Transition
from CustomEnv import CarlaEnv
import utils
import pickle




def carla_demo(exp):
    """
    DQfD_CartPole carla and record the measurements, the images, and the control signals
    :param
    :return:
    """
    demo_transitions=[]

    # file name format to save images
    out_filename_format = '_imageout/episode_{:0>4d}/{:s}/{:0>6d}'
    episode_reward = []

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
            print('Running at episode %d Frame %d' % (episode, frame))

            if not meas:
                action = None
            else:
                control = measurements.player_measurements.autopilot_control
                # print(control.steer)
                action_no, action = exp.action_discretize(control)
                # print(action.steer)
                actionprint = {
                    'action_number': action_no,
                    'steer': action.steer,
                    'throttle': action.throttle,
                    'brake': action.brake,
                    'Reverse': action.reverse
                }
                action_list.append(actionprint)
                print(actionprint)


            next_meas, next_state, reward, done, measurements = exp.step(action)
            next_state = utils.rgb_image_to_tensor(next_state['CameraRGB'])
            reward_list.append(reward)
            measurements_list.append(next_meas)


            # calculate and save reward into memory
            # Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'n_reward'))

            if meas:

                transition = Transition(meas,
                                        state,
                                        torch.tensor([[action_no]]),
                                        torch.tensor([[reward]]),
                                        next_state,
                                        next_meas,
                                        torch.zeros(1))   #TODO: use both the measurement and the image later
                demo_transitions.append(transition)

            # save image to disk
            for name, images in exp.cur_image.items():
                filename = out_filename_format.format(episode, name, frame)
                images.save_to_disk(filename)

            # Todo: remember to do the same in the self exploring part
            meas, state = next_meas, next_state

            # check for end condition
        #     if done:
        #         print('Target achieved!')
        #         break
        #
        # if not done:
        #     print("Target not achieved!")

        # save measurements, actions and rewards
        measurement_df = pd.DataFrame(measurements_list)
        measurement_df.to_csv('_measurements%d.csv' % episode)
        action_df = pd.DataFrame(action_list)
        action_df.to_csv('_actions%d.csv' % episode)
        reward_df = pd.DataFrame(reward_list)
        reward_df.to_csv('_reward%d.csv' % episode)
        episode_reward.append(sum(reward_list))

    print("Demonstration recorded! Average reward per episode:", sum(episode_reward)/config.CARLA_DEMO_EPISODE)

    return  demo_transitions, episode_reward



if __name__ == "__main__":

    exp = CarlaEnv(config.TARGET)
    exp.reset()
    demo_transitions, episode_reward = carla_demo(exp)

    with open(config.CARLA_DEMO_FILE, 'wb') as f:
        pickle.dump(demo_transitions, f)

    print("Recording finished!")
