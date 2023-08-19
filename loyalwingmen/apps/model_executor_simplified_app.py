
import sys
sys.path.append("..")

from datetime import datetime
import os

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3 import PPO
from modules.environments.drone_chase_env import DroneChaseEnv
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import callbacklist
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
import torch.nn as nn
import torch as th
import math


def main():
    
    

 
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\no_physics_in_1.00M_steps_reward_distance_low_frequency_speed_amplification\\models\\h[256, 256, 256]-f30-lr1e-08\\mPPO-r16211.7998046875-sd536.4000244140625.zip"
    
    model = PPO.load(selected_zip)
    env = DroneChaseEnv(GUI=True, rl_frequency=15, speed_amplification=1, debug=True)
    
    observation, info = env.reset()

    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward, action)

        if terminated:
            print("Episode terminated")
            env.reset()


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
