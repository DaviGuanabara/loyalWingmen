import time
import gym
from stable_baselines3 import PPO
from utils.callback_factory import gen_eval_callback, callbacklist
from utils.Logger import Logger
from envs.my_first_env import MyFirstEnv
from utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env

# from utils.keyboard_listener import KeyboardListener

# TODO Display text deve ser um objeto, para que pare de quebrar o terminal e eu não veja como está progredindo o treinamento.
# from utils.displaytext import logObs


from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

# Fazer um gerenciador de agentes e outro de obstáculo
# Fazer dentro do ambiente ou fora ?
train = True
test = False


if train:
    n_envs = 2

    env = make_vec_env(MyFirstEnv, n_envs=n_envs)
    eval_callback = callbacklist(
        env,
        log_path="./logs/",
        model_path="./models/",
        n_envs=n_envs,
        save_freq=10_000,
    )

    model = PPO(
        "MlpPolicy", env, verbose=0, device="auto"
    )  # + "/" str(learning_rate) +
    # reset_num_timesteps=False,
    model.learn(total_timesteps=10_000_000, callback=eval_callback)

if test:
    # keyboard_listener = KeyboardListener()
    model = PPO.load("./models/best_model.zip")
    env = MyFirstEnv(GUI=True)
    observation = env.reset()
    for steps in range(50_000):
        # button = keyboard_listener.get_action()

        # observation, reward, done, info = env.step(button)
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)

        logObs(observation, reward)
        # print(observation.velocity)
        if done:
            # print("done")
            env.reset()
