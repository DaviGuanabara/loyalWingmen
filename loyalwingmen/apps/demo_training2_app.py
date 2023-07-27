
import sys
sys.path.append("..")

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3 import PPO
from modules.environments.demo_env import DemoEnvironment
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
import torch.nn as nn
import torch as th
import math



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

    nn_t = [450, 247, 831]
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=dict(pi=nn_t, vf=nn_t)
    )
    # env = DemoEnvironment(GUI=False)

    model = PPO(
        CustomActorCriticPolicy,  # "CnnPolicy",
        vectorized_environment,
        verbose=0,
        device="auto",
        tensorboard_log="./logs/my_first_env/",
        policy_kwargs=policy_kwargs,
        learning_rate=math.pow(10, -7),
    )

    print(model.policy)
    model.learn(total_timesteps=1_000_000, callback=callback_list)
    model.save("demo_trained2_model")


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
