import os
import sys

sys.path.append("..")

import logging
import numpy as np

import pandas as pd
from scipy.stats import randint, uniform

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from modules.environments.demo_env import DemoEnvironment
from modules.models.policy import CustomActorCriticPolicy, CustomCNN


from modules.factories.callback_factory import gen_eval_callback, callbacklist
from modules.utils.Logger import Logger

from multiprocessing import cpu_count
from scipy.stats import randint

from datetime import datetime

# Configurando o logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def create_vectorized_environment(n_envs, frequency):
    env_fns = [DemoEnvironment for _ in range(n_envs)]
    vectorized_environment = SubprocVecEnv(env_fns)
    vectorized_environment.env_method("set_frequency", 240, frequency, indices=None)
    return vectorized_environment

def create_callback_list(n_envs, vectorized_environment):
    log_path = "./logs/"
    model_path = "./models/"
    save_freq = 100_000
    callback_list, storage_for_callback = callbacklist(
        vectorized_environment,
        log_path=log_path,
        model_path=model_path,
        n_envs=n_envs,
        save_freq=save_freq,
    )
    return callback_list, storage_for_callback

def create_policy_kwargs(hiddens, learning_rate):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=dict(pi=hiddens, vf=hiddens)
    )
    return policy_kwargs

def create_model(vectorized_environment, policy_kwargs, learning_rate):
    tensorboard_log = "./logs/my_first_env/"
    model = PPO(
        CustomActorCriticPolicy,
        vectorized_environment,
        verbose=0,
        device="auto",
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
    )
    return model

def save_model(model, hiddens, frequency, learning_rate):
    folder_name = f"model_{hiddens}_{frequency}_{learning_rate}"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models_{folder_name}_{current_time}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "my_model")
    model.save(model_path)
    print("Model saved at:", model_path)

def simulation(suggested):
    # Extrair os hiperparâmetros da lista sugerida
    hiddens, frequency, learning_rate = suggested

    number_of_logical_cores = cpu_count()
    n_envs = number_of_logical_cores

    vectorized_environment = create_vectorized_environment(n_envs, frequency)
    callback_list, storage_for_callback = create_callback_list(n_envs, vectorized_environment)
    policy_kwargs = create_policy_kwargs(hiddens, learning_rate)
    model = create_model(vectorized_environment, policy_kwargs, learning_rate)

    print(model.policy)
    model.learn(total_timesteps=1_000_000, callback=callback_list)

    save_model(model, hiddens, frequency, learning_rate)

    return storage_for_callback.best_mean_reward



# Configurando o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cross_validation_simulation(suggested, num_folds=5):
    """Executa a simulação várias vezes e retorna a média dos resultados."""
    scores = []
    for _ in range(num_folds):
        try:
            y = simulation(suggested)  
            scores.append(y)
        except Exception as e:
            logging.warning(f"Simulation failed with parameters {suggested}. Error: {e}")
    return np.mean(scores)

def generate_random_parameters():
    """Gera um conjunto aleatório de hiperparâmetros."""
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()  # Define o número de elementos em hiddens entre 1 e 8
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    # Frequência deve ser um valor inteiro múltiplo de 15 no intervalo de 0 a 120
    frequency = (randint(1, 8).rvs() * 15)  # Múltiplo de 15 no intervalo de 15 a 120
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return [hiddens, frequency, learning_rate]



def optimize_hyperparameters(num_iterations=100):
    """Otimiza os hiperparâmetros usando a otimização aleatória."""
    results = []
    best_score = float('-inf')
    stop_counter = 0

    for i in range(num_iterations):
        suggested = generate_random_parameters()
        avg_score = cross_validation_simulation(suggested, num_folds=5)

        results.append(suggested + [avg_score])

        logging.info(f"Iteration {i}, average score: {avg_score}, parameters: {suggested}")

        if avg_score > best_score:
            best_score = avg_score
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= 10:
            logging.info(f"Stopping early at iteration {i}")
            break

    return results

def save_results_to_excel(results):
    """Salva os resultados em um arquivo Excel."""
    df = pd.DataFrame(results, columns=['hiddens', 'frequency', 'learning_rate', 'score'])
    df['hiddens'] = df['hiddens'].apply(lambda x: ', '.join(map(str, x)))  # Converte listas em strings para salvar no Excel
    df.to_excel('simulation_results.xlsx', index=False)

def print_best_parameters(results):
    """Imprime os melhores parâmetros encontrados."""
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    print("Melhor pontuação: %.4f" % results[0][-1])
    print("Melhores parâmetros:")
    print("Hiddens: %s" % ', '.join(map(str, results[0][0])))
    print("Frequência: %.4f" % results[0][1])
    print("Taxa de aprendizado: %.10f" % results[0][2])  # Ajustada a precisão da taxa de aprendizado

def main():
    results = optimize_hyperparameters()
    save_results_to_excel(results)
    print_best_parameters(results)

if __name__ == "__main__":
    main()
