from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn
import torch as th
import math
from stable_baselines3 import PPO
from modules.environments.demo_env import DemoEnvironment
import sys
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import gen_eval_callback, callbacklist

sys.path.append("..")


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
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
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

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# ===============================================================================
# Setup
# ===============================================================================

def main():
    number_of_logical_cores = cpu_count()
    n_envs = number_of_logical_cores

    env_fns = []
    for _ in range(n_envs):
        env_fns.append(DemoEnvironment)

    vectorized_environment = SubprocVecEnv(env_fns)

    callback_list, storage_for_callback = callbacklist(
        vectorized_environment,
        log_path="./logs/",
        model_path="./models/",
        n_envs=n_envs,
        save_freq=100_000,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )
    # env = DemoEnvironment(GUI=False)

    model = PPO(
        "CnnPolicy",
        vectorized_environment,
        verbose=0,
        device="auto",
        tensorboard_log="./logs/my_first_env/",
        policy_kwargs=policy_kwargs,
        learning_rate=math.pow(10, -7),
    )

    model.learn(total_timesteps=3_000_000, callback=callback_list)
    model.save("demo_trained2_model")


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
