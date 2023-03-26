
import time
import gym
from stable_baselines3 import PPO

# from stable_baselines3.a2c import MlpPolicy



# from stable_baselines3.common.callbacks import ProgressBarCallback
# import ray
# from ray.tune import register_env
# from ray.rllib.agents import ppo

from utils.callback_factory import gen_eval_callback
from utils.Logger import Logger
from envs.my_first_env import my_first_env
from utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env


from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

#env = gym.make("my_first_env-v0")
env = my_first_env()
check_env(env)
#env = gym.make(my_first_env())
eval_callback = gen_eval_callback(
    env, "/log", "/models", eval_freq=1000)



start = time.time()

# model = PPO("MlpPolicy", env, verbose=1, device='auto', policy_kwargs=policy_kwargs, n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate, tensorboard_log=base_log_path) #+ topology_name) #+ "/" str(learning_rate) +
model = PPO("MlpPolicy", env, verbose=1, device='auto')  # + "/" str(learning_rate) +



model.learn(total_timesteps=10_000, reset_num_timesteps=False,
            callback=eval_callback) 

