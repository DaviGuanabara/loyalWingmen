
import sys
sys.path.append("..")

from datetime import datetime
import os

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3 import PPO
from modules.environments.drone_chase_env import DroneChaseEnv
from modules.environments.randomized_drone_chase_env import RandomizedDroneChaseEnv
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import callbacklist
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
import torch.nn as nn
import torch as th
import math


def main():
    
    

 
    #selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\1.00M_steps\\models\\h[1024, 1024, 1024]-f10-lr1e-06\\mPPO-r4674.5498046875-sd4209.28857421875.zip"
    #selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\4.00M_steps\\models\\h[256, 256, 256]-f10-lr1e-09\\best_model.zip"
    #selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\2.00M_steps\\models\\h[1024, 256, 2048]-f2-lr1e-06\\best_model.zip"
    #selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\2.00M_steps\\models\\h[1024, 256, 2048]-f2-lr1e-06\\mPPO-r16270.5498046875-sd4154.3701171875.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\1.50M_steps\\models\\h[512, 2048, 2048, 256]-f15-lr1e-05\\mPPO-r10641.7998046875-sd2006.248046875.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\DroneChaseEnv_no_physics_6.00M_steps_lidar_range_20m\\models\\h[512, 1024, 128, 256]-f1-lr0.0001\\best_model.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\DroneChaseEnv_no_physics_6.00M_steps_lidar_range_20m\\models\\h[512, 1024, 128, 256]-f1-lr0.0001\\mPPO-r9913.0400390625-sd2193.84765625.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\DroneChaseEnv_no_physics_6.00M_steps_lidar_range_20m\\models\\h[2048, 1024, 256, 2048]-f2-lr1e-06\\mPPO-r8023.93017578125-sd2374.717529296875.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\h[2048, 1024, 256, 2048]-f2-lr1e-06\\mPPO-r-15051.75-sd71.28314971923828.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\h[2048, 1024, 256, 2048]-f2-lr1e-06\\cima-para-baixo.zip"
    model = PPO.load(selected_zip)
    env = DroneChaseEnv(GUI=True, rl_frequency=30, speed_amplification=.2, debug=True)
    
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
