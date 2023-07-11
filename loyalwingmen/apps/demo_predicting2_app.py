import sys

sys.path.append("..")

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn
import torch as th
import math
from stable_baselines3 import PPO
from modules.environments.demo_env import DemoEnvironment




# ===============================================================================
# Execution
# ===============================================================================

model = PPO.load("demo_trained2_model")

env = DemoEnvironment(GUI=True)
observation, info = env.reset()

for steps in range(50_000):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    # env.show_lidar_log()

    if terminated:
        print("Episode terminated")
        env.reset()

        # I preferred to remove the reset to be able to make a long test
