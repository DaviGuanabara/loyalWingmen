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
import pandas as pd

# seed the pseudorandom number generator
from random import seed
from random import random, randint


class Training:
    def __init__(self):
        False

    def execute(self, topology, log_name, learning_rate=1e-5, n_repetitions=5):
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
                learning_rate=learning_rate,
            )

            model.learn(
                total_timesteps=1_0_000,
                callback=callback_list,
                tb_log_name="first_run",
            )

            rewards = np.append(rewards, storage_for_callback.best_mean_reward)

            # print(storage_for_callback)
        return np.mean(rewards)  # storage_for_callback


df = pd.DataFrame({"topology": [], "return": [], "index": [], "learning_rate": []})
df.to_excel("output.xlsx")
trainer = Training()
result_list = np.array([])

best_reward = -math.inf
best_topology = np.array([])

# TODO: variar também a taxa de aprendizagem
for i in range(500):
    # TODO a geração da topologia deveria estar em outra função
    topology = np.array([]).astype("int32")
    for _ in range(randint(2, 7)):  # número de camadas variáveis
        topology = np.append(topology, randint(16, 1000))
    # TODO achar nomes melhores.
    log_name = str(i)

    # TODO melhorar a geração do learning rate learning rate
    learning_rate = random() * math.pow(10, -randint(3, 10))

    # treinar o mesmo perfil 10 vezes para retirar uma análise estatística.
    # TODO esse bloco deve ser uma função que roda a execução 10 vezes e faz a média dos resultados.

    print(
        "Solution #" + str(i + 1) + " - ",
        "Topology:",
        np.array2string(topology),
        "Learning_rate:",
        learning_rate,
    )

    reward = trainer.execute(topology, log_name, learning_rate)

    print("Reward:", reward)

    # TODO aqui está a forma que eu vou salvar o treinamento. O mewlhor seria tamb[em salvar os logs e os modelos treinados de uma forma bonita. Enfim.

    df = pd.concat(
        # [df, pd.DataFrame([np.array2string(topology), reward, i, learning_rate])],
        [
            df,
            pd.DataFrame(
                {
                    "topology": [np.array2string(topology)],
                    "return": [reward],
                    "index": [i],
                    "learning_rate": [learning_rate],
                }
            ),
        ],
        ignore_index=True,
    )

    if best_reward < reward:
        best_reward = reward
        best_topology = topology
        print("Best reward and topology", best_reward, best_topology)

    print("Writing in output.xlsx in loyalwingmen folder")
    df.to_excel("output.xlsx")
    print("Dataframe wrote")

# Resultado, mostrando que deu tudo certo. O ideal seria eu salvar em algum arquivo txt, ou outra forma. Mas enfim, por enquanto será isso mesmo.
# TODO ajeitar a forma que é salva
print("Best reward and topology", best_reward, best_topology)

df.to_excel("conclusion.xlsx")
# melhor encontrado na primeira execu~c"ao que demorou 3 diasÇ Best reward and topology -7260.497177560001 [815 672 665 626 523 603]
# Best reward and topology -7260.497177560001 [815 672 665 626 523 603]
