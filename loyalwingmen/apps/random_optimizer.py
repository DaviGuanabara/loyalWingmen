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
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from modules.utils.Logger import Logger
from modules.environments.drone_and_cube_env import DroneAndCube
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
import math
import pandas as pd

# seed the pseudorandom number generator
from random import seed
from random import random, randint, choice


class Training:
    def __init__(self):
        False

    def execute(
        self,
        topology,
        log_name,
        learning_rate=1e-5,
        n_repetitions=2,
        total_rl_steps=2_000_000,
    ):
        rewards = np.array([])
        for i in range(n_repetitions):
            n_envs = 4
            nn_t = topology

            vectorized_environment = make_vec_env(DroneAndCube, n_envs=n_envs)
            # print(vectorized_environment.env_method("get_parameteres")[0])
            # aggregate_physics_steps = vectorized_environment.env_method("get_parameteres")[0].aggregate_physics_steps

            callback_list, storage_for_callback = callbacklist(
                vectorized_environment,
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
                vectorized_environment,
                verbose=0,
                device="auto",
                tensorboard_log="./logs/" + log_name + "/",
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
            )

            model.learn(
                total_timesteps=total_rl_steps,
                callback=callback_list,
                tb_log_name="first_run",
            )

            rewards = np.append(rewards, storage_for_callback.best_mean_reward)

            # print(storage_for_callback)
        return np.mean(rewards)  # storage_for_callback


def gen_topology(min_layers, max_layers):
    topology = np.array([]).astype("int32")
    for _ in range(randint(min_layers, max_layers)):  # número de camadas variáveis
        topology = np.append(topology, choice([256, 512, 1024]))  # 256, 512, 1024
    return topology


def save(df, topology, reward, index, learning_rate, total_rl_steps, n_repetitions):
    # TODO aqui está a forma que eu vou salvar o treinamento. O mewlhor seria tamb[em salvar os logs e os modelos treinados de uma forma bonita. Enfim.

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "topology": [np.array2string(topology)],
                    "return": [reward],
                    "index": [i],
                    "learning_rate": [learning_rate],
                    "total_timesteps": [total_rl_steps],
                    "n_repetitions": [n_repetitions],
                }
            ),
        ],
        ignore_index=True,
    )

    print("Writing in output.xlsx in loyalwingmen folder")
    df.to_excel("output.xlsx")
    print("Dataframe wrote")

    return df


def print_data(i, topology, learning_rate, reward):
    print(
        "Solution #" + str(i + 1) + " - ",
        "Topology:",
        np.array2string(topology),
        "Learning_rate:",
        learning_rate,
        "Reward:",
        reward,
    )


df = pd.DataFrame(
    {
        "topology": [],
        "return": [],
        "index": [],
        "learning_rate": [],
        "total_timesteps": [],
        "n_repetitions": [],
    }
)


df.to_excel("output.xlsx")
trainer = Training()
result_list = np.array([])

best_reward = -math.inf
best_topology = np.array([])

total_rl_steps = 500_000
n_repetitions = 1


for i in range(100):
    topology = gen_topology(3, 4)
    # TODO achar nomes melhores.
    log_name = str(i)
    learning_rate = 1 * math.pow(10, -randint(3, 15))

    reward = trainer.execute(
        topology,
        log_name,
        learning_rate,
        n_repetitions=n_repetitions,
        total_rl_steps=total_rl_steps,
    )

    print_data(i, topology, learning_rate, reward)
    df = save(df, topology, reward, i, learning_rate, total_rl_steps, n_repetitions)

    if best_reward < reward:
        best_reward = reward
        best_topology = topology
        print("New Best topology", best_topology, "\nReward:", best_reward)


print("Best topology", best_topology, "\nReward:", best_reward)
df.to_excel("conclusion.xlsx")
