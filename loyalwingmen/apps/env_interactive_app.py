import time
import os
import sys
sys.path.append("..")
from stable_baselines3 import PPO
from modules.factories.callback_factory import callbacklist
from modules.utils.Logger import Logger
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import math


from modules.utils.keyboard_listener import KeyboardListener
from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env
from pathlib import Path
from modules.environments.drone_chase_level1 import DroneChaseEnvLevel1
import logging
import numpy as np

debug = True

env = DroneChaseEnvLevel1(GUI=True, rl_frequency=15, debug=debug)
keyboard_listener = KeyboardListener(env.get_keymap())


observation, info = env.reset()
for steps in range(50_000):
    action = keyboard_listener.get_action()
    action = np.array([-0.14996707, -0.44416678, -0.5607476,   0.7638135 ])
    observation, reward, terminated, truncated, info = env.step(action)
    
    
    #logging.debug(f"(main) reward: {reward}")
    print(f"reward:{reward:.2f} - action:{action} - observation:{observation}, ")
    if terminated:
        observation, info = env.reset()