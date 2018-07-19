# -*- coding: utf-8 -*
"""
"""

import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import config
from IPython import display


def rgb_image_to_tensor(camera_rgb):
    """
    convert RGB image from Carla to tensor using height=CARLA_IMG_HEIGHT
    :param camera_rgb:
    :return:
    """
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((config.CARLA_IMG_HEIGHT, config.CARLA_IMG_WIDTH), interpolation=Image.CUBIC),
                        T.ToTensor()])
    s = np.ascontiguousarray(camera_rgb.data, dtype=np.float32) / 255
    s = torch.from_numpy(s)
    # Resize, and add a batch dimension (BCHW)
    return resize(s).unsqueeze(0)


def plot_rgb_image(camera_rgb):
    """
    Plot an RGB image from Carla
    :param camera_rgb:
    :return:
    """
    plt.figure()
    plt.imshow(camera_rgb.data)
    plt.show()



def plot_reward(y):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(y)
    # plt.pause(0.001)  # pause a bit so that plots are updated


