import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from openpyxl import Workbook
from datetime import datetime

sys.path.append("..")

from modules.environments.demo_env import DemoEnvironment
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from multiprocessing import cpu_count
from typing import List

# Configurando o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_vectorized_environment(n_envs: int, frequency: int) -> SubprocVecEnv:
    env_fns = [DemoEnvironment for _ in range(n_envs)]
    vectorized_environment = SubprocVecEnv(env_fns)
    vectorized_environment.env_method("set_frequency", 240, frequency, indices=None)
    return vectorized_environment

def create_callback_list(n_envs: int, vectorized_environment: SubprocVecEnv, callbacks_to_include: List[str]):
    log_path = "./logs/"
    model_path = "./models/"
    save_freq = 100_000
    callback_list, storage_for_callback = callbacklist(
        vectorized_environment,
        log_path=log_path,
        model_path=model_path,
        n_envs=n_envs,
        save_freq=save_freq,
        callbacks_to_include=callbacks_to_include
    )
    return callback_list, storage_for_callback

def create_policy_kwargs(hiddens: list, learning_rate: float) -> dict:
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=dict(pi=hiddens, vf=hiddens)
    )
    return policy_kwargs

def create_model(vectorized_environment: SubprocVecEnv, policy_kwargs: dict, learning_rate: float) -> PPO:
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

def train_model(model: PPO, callback_list, n_timesteps: int = 1_000_000) -> PPO:
    model.learn(total_timesteps=n_timesteps, callback=callback_list)
    return model

def save_model(model: PPO, hiddens: list, frequency: int, learning_rate: float):
    folder_name = f"model_{hiddens}_{frequency}_{learning_rate}"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models_{folder_name}_{current_time}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "my_model")
    model.save(model_path)
    logging.info(f"Model saved at: {model_path}")

def cross_validation_simulation(hiddens: list, frequency: int, learning_rate: float, num_folds: int) -> float:
    number_of_logical_cores = cpu_count()
    n_envs = number_of_logical_cores

    vectorized_environment = create_vectorized_environment(n_envs, frequency)
    vectorized_environment = VecMonitor(vectorized_environment)
    
    callback_list, _ = create_callback_list(n_envs, vectorized_environment, callbacks_to_include=["progressbar"])
    policy_kwargs = create_policy_kwargs(hiddens, learning_rate)
    model = create_model(vectorized_environment, policy_kwargs, learning_rate)

    logging.info(model.policy)
    model = train_model(model, callback_list)
    save_model(model, hiddens, frequency, learning_rate)
    avg_score, std_dev = evaluate_policy(model, vectorized_environment, n_eval_episodes=num_folds, deterministic=True)
    print("avg_score", avg_score)
    return avg_score  # we only return the average score, not the standard deviation


def generate_random_parameters() -> list:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate

def print_iteration_results(iteration: int, avg_score: float, hiddens: list, frequency: int, learning_rate: float):
    print(f"Iteration {iteration} results:")
    print("Best score: %.4f" % avg_score)
    print("Best parameters:")
    print("Hiddens: %s" % ', '.join(map(str, hiddens)))
    print("Frequency: %.4f" % frequency)
    print("Learning rate: %.10f" % learning_rate)

def optimize_hyperparameters(num_iterations: int = 100) -> list:
    results = []
    best_score = float('-inf')
    stop_counter = 0

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(['Iteration', 'Hiddens', 'Frequency', 'Learning Rate', 'Score'])

    for i in range(num_iterations):
        hiddens, frequency, learning_rate = generate_random_parameters()
        avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_folds=5)

        result = (hiddens, frequency, learning_rate, avg_score)
        results.append(result)
        logging.info(f"Iteration {i}, average score: {avg_score}, parameters: {result}")

        print_iteration_results(i, avg_score, hiddens, frequency, learning_rate)

        worksheet.append([i, ', '.join(map(str, hiddens)), frequency, learning_rate, avg_score])
        workbook.save('simulation_results.xlsx')

    return results

def main():
    results = optimize_hyperparameters()
    print(results)

if __name__ == "__main__":
    main()
