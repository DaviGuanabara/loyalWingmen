import os
import sys
sys.path.append("..")
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from modules.environments.demo_env import DemoEnvironment
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import gen_eval_callback, callbacklist
from typing import List, Tuple
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

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

def save_model(model: PPO, model_path: str):
    model.save(model_path)
    logging.info(f"Model saved at: {model_path}")

def cross_validation_simulation(hiddens: list, frequency: int, learning_rate: float, num_folds: int) -> float:
    number_of_logical_cores = os.cpu_count()
    n_envs = number_of_logical_cores

    vectorized_environment = create_vectorized_environment(n_envs, frequency)
    vectorized_environment = VecMonitor(vectorized_environment)
    
    callback_list, _ = create_callback_list(n_envs, vectorized_environment, callbacks_to_include=["progressbar"])
    policy_kwargs = create_policy_kwargs(hiddens, learning_rate)
    model = create_model(vectorized_environment, policy_kwargs, learning_rate)

    logging.info(model.policy)
    model = train_model(model, callback_list)
    avg_score, std_dev = evaluate_policy(model, vectorized_environment, n_eval_episodes=num_folds, deterministic=True)
    logging.info(f"Avg score: {avg_score}")
    return avg_score

def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate

def objective(trial: Trial) -> float:
    hiddens = trial.suggest_int('hiddens', 10, 1000)
    num_hiddens = trial.suggest_int('num_hiddens', 3, 8)
    hiddens = [trial.suggest_int(f'hiddens_{i}', 10, 1000) for i in range(num_hiddens)]
    frequency = trial.suggest_int('frequency', 15, 120)
    learning_rate = trial.suggest_float('learning_rate', 0.00000000001, 0.1)
    
    #TODO melhorar o nome num_folds. 
    #TODO melhorar o nome cross_validation_simulation
    avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_folds=100)
    return avg_score

def optimize_hyperparameters(num_iterations: int = 100) -> List[Tuple[List[int], int, float, float]]:
    """Optimize hyperparameters using the Bayesian optimization algorithm.

    Parameters
    ----------
    num_iterations : int
        The number of iterations to run the algorithm for.

    Returns
    -------
    results : List[Tuple[List[int], int, float, float]]
        The results of each iteration, as a list of tuples containing the hiddens, frequency, learning_rate, and score.
    """

    results = []
    best_score = float('-inf')
    stop_counter = 0

    # DataFrame para armazenar os resultados
    df = pd.DataFrame(columns=['hiddens', 'frequency', 'learning_rate', 'score'])

    for i in range(num_iterations):
        hiddens, frequency, learning_rate = generate_random_parameters()
        avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_folds=5)

        results.append((hiddens, frequency, learning_rate, avg_score))
        logging.info(f"Iteration {i}, average score: {avg_score}, parameters: {hiddens, frequency, learning_rate}")

        print_iteration_results(i, avg_score, hiddens, frequency, learning_rate)

        df.loc[i] = [hiddens, frequency, learning_rate, avg_score]

        folder_path = os.path.join("results-bayesian-optimization", 'simulation_results.xlsx')
        with pd.ExcelWriter(folder_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False, header=not writer.sheets['Results'])

        if avg_score > best_score:
            best_score = avg_score
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter >= 10:
            logging.info(f"Stopping early at iteration {i}")
            break

    return results

def print_iteration_results(iteration: int, avg_score: float, hiddens: List[int], frequency: int, learning_rate: float):
    print(f"Iteration {iteration} results:")
    print("Best score: %.4f" % avg_score)
    print("Best parameters:")
    print("Hiddens: %s" % ', '.join(map(str, hiddens)))
    print("Frequency: %.4f" % frequency)
    print("Learning rate: %.10f" % learning_rate)

def save_results_to_excel(results: List[Tuple[List[int], int, float, float]]):
    df = pd.DataFrame(results, columns=['hiddens', 'frequency', 'learning_rate', 'score'])
    df['hiddens'] = df['hiddens'].apply(lambda x: ', '.join(map(str, x)))
    folder_path = os.path.join("results-bayesian-optimization", 'simulation_results.xlsx')
    df.to_excel(folder_path, index=False)

def print_best_parameters(results: List[Tuple[List[int], int, float, float]]):
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    print("Best score: %.4f" % results[0][-1])
    print("Best parameters:")
    print("Hiddens: %s" % ', '.join(map(str, results[0][0])))
    print("Frequency: %.4f" % results[0][1])
    print("Learning rate: %.10f" % results[0][2])    

def main():
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=100)

    results = []
    for trial in study.trials:
        hiddens = [trial.params[f'hiddens_{i}'] for i in range(trial.params['num_hiddens'])]
        results.append((hiddens, trial.params['frequency'], trial.params['learning_rate'], trial.value))
        print("saving")
        save_results_to_excel(results)
        print("saved")
        print("printing")
        print_best_parameters(results)
        print("printed")

    
    

if __name__ == "__main__":
    main()
