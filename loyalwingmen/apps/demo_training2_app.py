import sys

sys.path.append("..")

from modules.environments.demo_env import DemoEnvironment

from stable_baselines3 import PPO
import math

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# from multiprocessing import cpu_count
# from stable_baselines3.common.vec_env import SubprocVecEnv

"""
Demo.app is a file made to show a simple execution of an environment.
I hope you enjoy it.
Problem:
It is necessary to highlight that KeyboardListener() won't work in latest MacOS:

'Recent versions of macOS restrict monitoring of the keyboard for security reasons. For that reason, one of the following must be true:

1. The process must run as root.
3. Your application must be white listed under Enable access for assistive devices. 
   Note that this might require that you package your application, since otherwise the entire Python installation must be white listed.
2. On versions after Mojave, you may also need to whitelist your terminal application if running your script from a terminal.'

More informations in: https://pynput.readthedocs.io/en/latest/limitations.html

I were not able to make it work in MacOS Ventura on M1 Pro
"""

# ===============================================================================
# Veritifation
# ===============================================================================


# ===============================================================================
# Setup
# ===============================================================================
# number_of_logical_cores = cpu_count()
# n_envs = number_of_logical_cores

# env_fns = []
# for _ in range(n_envs):
#    env_fns.append(DemoEnvironment)


# vectorized_environment = SubprocVecEnv(env_fns)
# é preciso colocar o if __main__ lá em vaixo para funcionar

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False,
)
env = DemoEnvironment(GUI=False)

model = PPO(
    "CnnPolicy",
    env,
    verbose=0,
    device="auto",
    tensorboard_log="./logs/my_first_env/",
    # policy_kwargs=[256, 256, 256],
    policy_kwargs=policy_kwargs,
    learning_rate=math.pow(10, -5),
)

model.learn(total_timesteps=25000)
model.save("demo_trained_model")

# ===============================================================================
# Execution
# ===============================================================================
del model  # remove to demonstrate saving and loading
model = PPO.load("demo_trained_model")

env = DemoEnvironment(GUI=True)
observation, info = env.reset()

for steps in range(50_000):
    action = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    # env.show_lidar_log()

    if terminated:
        print("Episode terminated")

        # I preferred to remove the reset to be able to make a long test
        # env.reset()
