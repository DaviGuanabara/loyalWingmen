
import time
import gym
from stable_baselines3 import PPO

# from stable_baselines3.a2c import MlpPolicy



# from stable_baselines3.common.callbacks import ProgressBarCallback
# import ray
# from ray.tune import register_env
# from ray.rllib.agents import ppo

from utils.callback_factory import gen_eval_callback, callbacklist
from utils.Logger import Logger
from envs.my_first_env import MyFirstEnv
from utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env


from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

train = False
test = True


if train:

    n_envs = 2

    env = make_vec_env(MyFirstEnv, n_envs=n_envs)
    eval_callback = callbacklist(env, log_path="./logs/", model_path="./models/",
                                n_envs=n_envs, save_freq=10_000)

    model = PPO("MlpPolicy", env, verbose=0, device='auto')  # + "/" str(learning_rate) +
    # reset_num_timesteps=False,
    model.learn(total_timesteps=10_000, callback=eval_callback)

if test:     

    env = MyFirstEnv(GUI=True)
    for steps in range(5000):
        # agent policy that uses the observation and info
        #action = env.action_space.sample()
        observation, reward, done, info = env.step([])

        if done:
            #print("done")
            env.reset()
