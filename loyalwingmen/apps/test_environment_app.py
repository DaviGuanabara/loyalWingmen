import time
import os
import sys
from stable_baselines3 import PPO
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from modules.utils.Logger import Logger
from modules.environments.drone_and_cube_env import DroneAndCube
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import math


from modules.utils.keyboard_listener import KeyboardListener
from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env
from modules.environments.loyalwingmen.lidar_env import DroneLidar


env = DroneLidar(GUI=True)

# funciona para sb3 a partir de 2.0.0
observation, info = env.reset()
keyboard_listener = KeyboardListener()
for steps in range(50_000):
    action = keyboard_listener.get_action(intensity=0.05)
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation)
    # TODO: display text e logreturn pode ser incorporado pelo ambiente.
    env.show_lidar_log()

    # log_returns(observation, reward, action)
    # if terminated:
    # observation, info = env.reset()
