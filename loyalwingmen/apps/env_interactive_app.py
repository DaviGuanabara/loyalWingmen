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
from modules.environments.level1_environment import Level1
import logging
import numpy as np

debug = True

env = Level1(GUI=True, rl_frequency=15, debug=debug)
# check_env(env)
print("env created")

# keyboard_listener = KeyboardListener(env.get_keymap())
print("keyboard_listener created")

print("waiting for reset")
observation, info = env.reset()
print(f"observation:{observation}, ")
for _ in range(50_000):
    print("waiting for action")
    # action = keyboard_listener.get_action()
    # print("action received", action)
    action = np.array([0.1, 0.1, 0.1, 0.1])
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"reward:{reward:.2f} - action:{action} - observation:{observation}, ")

    if terminated:
        observation, info = env.reset()
