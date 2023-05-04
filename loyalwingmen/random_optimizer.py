"""
Let f: ℝn → ℝ be the fitness or cost function which must be minimized. Let x ∈ ℝn designate a position or candidate solution in the search-space. The basic RO algorithm can then be described as:

Initialize x with a random position in the search-space.
Until a termination criterion is met (e.g. number of iterations performed, or adequate fitness reached), repeat the following:
Sample a new position y by adding a normally distributed random vector to the current position x
If (f(y) < f(x)) then move to the new position by setting x = y
Now x holds the best-found position.
This algorithm corresponds to a (1+1) evolution strategy with constant step-size.
"""

"""
É aí que está. Tenho que fazer um algoritmo de treinamento no qual ele recebe os parâmetros de treinamento e retorna o melhor valor obtido.
"""
from stable_baselines3.common.env_util import make_vec_env
import gym
from stable_baselines3 import PPO
from utils.factories.callback_factory import gen_eval_callback, callbacklist
from utils.Logger import Logger
from envs.my_first_env import MyFirstEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
import math

# seed the pseudorandom number generator
from random import seed
from random import random, randint


class Training:
    def __init__(self):
        False

    def execute(self, topology, log_name, n_repetitions=10):
        rewards = np.array([])
        for i in range(n_repetitions):
            n_envs = 4
            nn_t = topology

            env = make_vec_env(MyFirstEnv, n_envs=n_envs)
            callback_list, storage_for_callback = callbacklist(
                env,
                log_path="./logs/",
                model_path="./models/",
                n_envs=n_envs,
                save_freq=100_000,
            )

            # TODO isso daqui tá dando pau. Então, ajeitar
            # UserWarning: As shared layers in the mlp_extractor are removed since SB3 v1.8.0,
            # you should now pass directly a dictionary and not a list (net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])
            policy_kwargs = dict(
                activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=nn_t, vf=nn_t)
            )

            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                device="auto",
                tensorboard_log="./logs/" + log_name + "/",
                policy_kwargs=policy_kwargs,
            )

            model.learn(
                total_timesteps=500_000, callback=callback_list, tb_log_name="first_run"
            )

            rewards = np.append(rewards, storage_for_callback.best_mean_reward)

            # print(storage_for_callback)
        return np.mean(rewards)  # storage_for_callback


trainer = Training()
result_list = np.array([])

best_reward = -math.inf
best_topology = np.array([])

for i in range(10):
    # TODO a geração da topologia deveria estar em outra função
    topology = np.array([]).astype("int32")
    for _ in range(randint(3, 10)):  # número de camadas variáveis
        topology = np.append(topology, randint(16, 1000))
    # TODO achar nomes melhores.
    log_name = str(i)

    # treinar o mesmo perfil 10 vezes para retirar uma análise estatística.
    # TODO esse bloco deve ser uma função que roda a execução 10 vezes e faz a média dos resultados.
    reward = trainer.execute(topology, log_name)

    if best_reward < reward:
        best_reward = reward
        best_topology = topology
        print("Best reward and topology", best_reward, best_topology)

# Resultado, mostrando que deu tudo certo. O ideal seria eu salvar em algum arquivo txt, ou outra forma. Mas enfim, por enquanto será isso mesmo.
# TODO ajeitar a forma que é salva
print("Best reward and topology", best_reward, best_topology)

# melhor encontrado na primeira execu~c"ao que demorou 3 diasÇ Best reward and topology -7260.497177560001 [815 672 665 626 523 603]
# Best reward and topology -7260.497177560001 [815 672 665 626 523 603]
