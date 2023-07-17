import os
import sys
import logging
import numpy as np
import pandas as pd
import optuna
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from modules.environments.demo_env import DemoEnvironment
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from multiprocessing import cpu_count
from datetime import datetime
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

    logging.info(f"Policy architecture: {model.policy}")
    model = train_model(model, callback_list)
    save_model(model, hiddens, frequency, learning_rate)
    avg_score, std_dev = evaluate_policy(model, vectorized_environment, n_eval_episodes=num_folds, deterministic=True)
    return avg_score  # Retorna apenas a pontuação média, não o desvio padrão

def generate_random_parameters(trial: optuna.Trial) -> tuple:
    hidden_dist = trial.suggest_int('hidden_dist', 10, 1000)
    num_hiddens = trial.suggest_int('num_hiddens', 3, 8)
    hiddens = [trial.suggest_int(f'hidden_{i}', 10, 1000) for i in range(num_hiddens)]
    frequency = trial.suggest_int('frequency', 1, 8) * 15
    learning_rate = trial.suggest_uniform('learning_rate', 0.00000000001, 0.1)
    return hiddens, frequency, learning_rate

def optimize_hyperparameters(num_iterations: int = 100) -> pd.DataFrame:
    study = optuna.create_study(direction='maximize')

    def objective(trial: optuna.Trial):
        hiddens, frequency, learning_rate = generate_random_parameters(trial)
        avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_folds=5)
        return avg_score

    study.optimize(objective, n_trials=num_iterations)

    return study.trials_dataframe()

def main():
    results = optimize_hyperparameters()
    print(results)

if __name__ == "__main__":
    main()
