import time
import os
import sys
from stable_baselines3 import PPO
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from modules.utils.Logger import Logger
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import math


from modules.utils.keyboard_listener import KeyboardListener
from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env
from pathlib import Path


def generate_model(vectorized_enviroment, nn_t=[512, 512, 512]):
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=nn_t, vf=nn_t))

    model = PPO(
        "MlpPolicy",
        vectorized_enviroment,
        verbose=0,
        device="auto",
        tensorboard_log="./logs/my_first_env/",
        policy_kwargs=policy_kwargs,
        learning_rate=math.pow(10, -5),
    )

    return model


def train(
    model, vectorized_environment, total_rl_steps=300_000, n_envs=4, save_freq=100_000
):
    total_rl_steps = 300_000
    n_envs = 4

    callback, _ = callbacklist(
        vectorized_environment,
        log_path="./logs/",
        model_path="./models/",
        n_envs=n_envs,
        save_freq=save_freq,
    )

    model.learn(
        total_timesteps=total_rl_steps, callback=callback, tb_log_name="first_run"
    )


def test(model, environment_function):
    base_path = str(Path(os.getcwd()).parent.absolute())

    # assert os.path.isfile(
    #    base_path + "/models/best_model.zip"
    # ), "There isn't 'best model' available to test"

    model = PPO.load("./models/best_model")

    # loyalwingmen\apps\models\best_model.zip

    keyboard_listener = KeyboardListener()
    env = environment_function(GUI=True)

    # funciona para sb3 a partir de 2.0.0
    observation, info = env.reset()
    for steps in range(50_000):
        # action = keyboard_listener.get_action()

        action, _states = model.predict(observation, deterministic=True)
        # observation, reward, done, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)

        # TODO: display text e logreturn pode ser incorporado pelo ambiente.
        env.show_log()
        # log_returns(observation, reward, action)
        if terminated:
            observation, info = env.reset()


assert len(sys.argv) - 1 > 0, "Please, add arguments, as 'test' or 'train'."

train_flag = True if "train" in sys.argv else False  # False
test_flag = True if "test" in sys.argv else False  # True


if train_flag:
    vectorized_environment = make_vec_env(DroneAndCube, n_envs=4)
    model = generate_model(vectorized_environment, nn_t=[512, 512, 512])
    train(
        model,
        vectorized_environment,
        total_rl_steps=300_000,
        n_envs=4,
        save_freq=100_000,
    )

if test_flag:
    assert os.path.isfile(
        "./models/best_model.zip"
    ), "There isn't 'best model' available to test"

    model = PPO.load("./models/best_model")

    test(model, DroneAndCube)
