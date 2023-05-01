from stable_baselines3.common.env_util import make_vec_env
import gym
from stable_baselines3 import PPO
from utils.factories.callback_factory import gen_eval_callback, callbacklist
from utils.Logger import Logger
from envs.my_first_env import MyFirstEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
import math

# seed the pseudorandom number generator
from random import seed
from random import random, randint


# -b +- raiz de (b2 - 4ac) / 2a -> (-b +- b) / 2a : 0, 1 (-b/a)
# x = 0.5 is max
def evaluation(x):
    return -math.pow(x, 2) + x


def random_optimizer():
    best_value = -math.inf
    best_x = None
    for _ in range(100):
        x = random()
        value = evaluation(x)

        if value > best_value:
            best_value = value
            best_x = x

    return best_x, best_value


print(random_optimizer())
