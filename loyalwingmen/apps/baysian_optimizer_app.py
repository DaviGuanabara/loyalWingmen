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
from modules.factories.callback_factory import callbacklist, CallbackType
from typing import List, Tuple
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import warnings
from gymnasium import Env
from gymnasium import spaces, Env
from typing import Optional, Union
from ml.pipeline import ReinforcementLearningPipeline
from ml.directory_manager import DirectoryManager


#warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    



    

def cross_validation_simulation(hiddens: list, frequency: int, learning_rate: float, num_evaluations: int, output_folder: str, n_timesteps: int) -> float:
    print(f"Running simulation with hiddens={hiddens}, frequency={frequency}, learning_rate={learning_rate}")
    number_of_logical_cores = os.cpu_count()
    n_envs: int = number_of_logical_cores if number_of_logical_cores is not None else 1

    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(n_envs, frequency)

    model_dir = os.path.join(output_folder, 'models')
    log_dir = os.path.join(output_folder, 'logs')
    callback_list= ReinforcementLearningPipeline.create_callback_list(n_envs, vectorized_environment, model_dir=model_dir, log_dir=log_dir, callbacks_to_include=[CallbackType.PROGRESSBAR])
    policy_kwargs = ReinforcementLearningPipeline.create_policy_kwargs(hiddens, learning_rate)
    model = ReinforcementLearningPipeline.create_model(vectorized_environment, policy_kwargs, learning_rate)

    logging.info(model.policy)
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps)
    
    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(model, vectorized_environment)
    ReinforcementLearningPipeline.save_model(model, hiddens, frequency, learning_rate, avg_reward, std_dev, output_folder)
    
     
    #evaluate_with_dynamic_episodes(model, vectorized_environment)
    logging.info(f"Avg score: {avg_reward}")
    return avg_reward

def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate




def objective(trial: Trial, output_folder: str, n_timesteps: int, study_name: str) -> float:
    
    num_hiddens = trial.suggest_int('num_hiddens', 3, 4)
    hiddens = [trial.suggest_int(f'hiddens_{i}', 100, 1000) for i in range(num_hiddens)]
    frequency = trial.suggest_int('frequency', 1, 2) * 15
    exponent = trial.suggest_float('exponent', -10, -1)
    learning_rate = 10 ** exponent

    
    print("Parameters:")
    print("Hiddens: ", hiddens)
    print("Frequency: ", frequency)
    print("Learning Rate: ", learning_rate)
    

    avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_evaluations=100, output_folder=output_folder, n_timesteps=n_timesteps)
    print(avg_score)
    # Salvar os resultados na planilha
    result = (hiddens, frequency, learning_rate, avg_score)
    print("saving results")
    file_name = f"results_{study_name}.xlsx"
    ReinforcementLearningPipeline.save_results_to_excel(output_folder, file_name, [result])
    print("results saved")

    return avg_score

def print_best_parameters(results: List[Tuple[List[int], int, float, float]]):
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    print("Best score: %.4f" % results[0][-1])
    print("Best parameters:")
    print("Hiddens: %s" % ', '.join(map(str, results[0][0])))
    print("Frequency: %.4f" % results[0][1])
    print("Learning rate: %.10f" % results[0][2])

def main():
    
    
    n_timesteps = 1_000_000
    experiment_name = "Optimizer_baysian_app"
    study_name = "no_physics"
    # Criar a pasta de saída
    output_folder = DirectoryManager.create_output_folder(experiment_name)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name=study_name, storage=f'sqlite:///{output_folder}/optimizer.db')
    study.optimize(lambda trial: objective(trial, output_folder, n_timesteps, study_name), n_trials=100)

    results = []
    for trial in study.trials:
        hiddens = [trial.params[f'hiddens_{i}'] for i in range(trial.params['num_hiddens'])]
        results.append((hiddens, trial.params['frequency'], trial.params['learning_rate'], trial.value))

    print_best_parameters(results)

if __name__ == "__main__":
    main()
