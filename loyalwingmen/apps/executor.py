import time
import os
import sys
sys.path.append("..")
from stable_baselines3 import PPO, SAC
from modules.factories.callback_factory import callbacklist
from modules.utils.Logger import Logger
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from modules.environments.drone_chase_level1 import DroneChaseEnvLevel1

#policy_kwargs = dict(
#                net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
            #)

env = DroneChaseEnvLevel1(GUI=True, rl_frequency=1, debug=True)

model = SAC(
                "MlpPolicy",
                env,
                verbose=0,
                device="cuda",
            )


model.load("./sac_drone_chase_level1")
observation, info = env.reset()
for steps in range(50_000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    
    
    #logging.debug(f"(main) reward: {reward}")
    print(f"reward:{reward:.2f} - action:{action}, ")
    if terminated:
        observation, info = env.reset()