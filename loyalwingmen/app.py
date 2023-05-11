import time
import gym
from stable_baselines3 import PPO
from utils.factories.callback_factory import gen_eval_callback, callbacklist
from utils.Logger import Logger
from envs.my_first_env import MyFirstEnv
from utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th

# TODO está treinando. Mas falta adicionar o cubo e ajeitar o código.
from utils.keyboard_listener import KeyboardListener

# TODO Display text deve ser um objeto, para que pare de quebrar o terminal e eu não veja como está progredindo o treinamento.
from utils.displaytext import log_returns

# TODO Aplicar o random search. Basicamente, iterar o "train" com parâmetros aleatórios, e pegar o maior.

from typing import Callable
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

# TODO criar uma pasta com OUTPUTS, contendo logs, models, por experimento.
# Fazer um gerenciador de agentes e outro de obstáculo
# Fazer dentro do ambiente ou fora ?
train = True
test = False


if train:
    n_envs = 4

    env = make_vec_env(MyFirstEnv, n_envs=n_envs)
    callback, _ = callbacklist(
        env,
        log_path="./logs/",
        model_path="./models/",
        n_envs=n_envs,
        save_freq=100_000,
    )

    nn_t = [815, 672, 665, 626, 523, 603]  # Valor tirado do otimizador aleatório
    # nn_t = [256, 256, 256]
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=nn_t, vf=nn_t))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="auto",
        tensorboard_log="./logs/my_first_env/",
        policy_kwargs=policy_kwargs,
    )  # + "/" str(learning_rate) +
    # reset_num_timesteps=False,
    model.learn(total_timesteps=1_000_000, callback=callback, tb_log_name="first_run")

if test:
    keyboard_listener = KeyboardListener()
    model = PPO.load("./models/best_model.zip")
    env = MyFirstEnv(GUI=True)
    observation = env.reset()
    for steps in range(50_000):
        action = keyboard_listener.get_action()

        # action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)

        log_returns(observation, reward, action)
        # time.sleep(1)
        # print(observation.velocity)
        if done:
            # print("done")
            env.reset()
