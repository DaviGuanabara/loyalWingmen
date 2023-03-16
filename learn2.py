"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import callback_factory

DEFAULT_RLLIB = False
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


env = make_vec_env("takeoff-aviary-v0", n_envs=4, monitor_dir="logs")
print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

callbacklist = callback_factory.callbacklist(env, n_envs=4)


model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000, callback=callbacklist)

#### Show (and record a video of) the model's performance ##
env = TakeoffAviary(gui=DEFAULT_GUI, record=DEFAULT_RECORD_VIDEO)

obs = env.reset()
start = time.time()
for i in range(3*env.SIM_FREQ):
    
    action, _states = model.predict(obs, deterministic=True)



    obs, reward, done, info = env.step(action)

    if i % env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    if done:
        obs = env.reset()

env.close()
