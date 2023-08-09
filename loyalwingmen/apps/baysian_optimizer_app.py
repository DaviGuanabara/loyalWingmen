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
from apps.ml.pipeline import ReinforcementLearningPipeline
from apps.ml.directory_manager import DirectoryManager
import torch as th
from torch import backends
from sys import platform

#warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    



    

def cross_validation_simulation(hiddens: list, frequency: int, learning_rate: float, n_timesteps: int, models_dir: str, logs_dir: str) -> float:
    print(f"Running simulation with hiddens={hiddens}, frequency={frequency}, learning_rate={learning_rate}")
    number_of_logical_cores = os.cpu_count()
    n_envs: int = number_of_logical_cores if number_of_logical_cores is not None else 1
    #n_envs = int(n_envs / 2)
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(n_envs, frequency)
    
    
    callback_list= ReinforcementLearningPipeline.create_callback_list(n_envs, vectorized_environment, model_dir=models_dir, log_dir=logs_dir, callbacks_to_include=[CallbackType.PROGRESSBAR])
    policy_kwargs = ReinforcementLearningPipeline.create_policy_kwargs(hiddens, learning_rate)
    model = ReinforcementLearningPipeline.create_model(vectorized_environment, policy_kwargs, learning_rate)

    logging.info(model.policy)
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps)
    
    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(model, vectorized_environment)
    ReinforcementLearningPipeline.save_model(model, hiddens, frequency, learning_rate, avg_reward, std_dev, models_dir)
    
    return avg_reward

def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate


def suggest_parameters(trial: Trial) -> Tuple[list, int, float]:
    num_hiddens = trial.suggest_int('num_hiddens', 3, 4)
    hiddens = [trial.suggest_int(f'hiddens_{i}', 100, 1000) for i in range(num_hiddens)]
    frequency = trial.suggest_int('frequency', 1, 2) * 15
    exponent = trial.suggest_float('exponent', -10, -1)
    learning_rate = 10 ** exponent
    
    logging.info(
        f"[suggest_parameters] Suggested Parameters:\n"
        f"  - Hiddens: {', '.join(map(str, hiddens))}\n"
        f"  - Frequency: {frequency}\n"
        f"  - Learning Rate: {learning_rate:.10f}"
    )
    
    return hiddens, frequency, learning_rate


def objective(trial: Trial, output_folder: str, n_timesteps: int, study_name: str, models_dir: str, logs_dir:str) -> float:
    
    hiddens, frequency, learning_rate = suggest_parameters(trial)
    
    avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, models_dir=models_dir, logs_dir=logs_dir, n_timesteps=n_timesteps)
    logging.info(f"Avg score: {avg_score}")

    print("saving results...")
    result = (hiddens, frequency, learning_rate, avg_score)
    ReinforcementLearningPipeline.save_results_to_excel(output_folder, f"results_{study_name}.xlsx", [result])
    print("results saved")

    return avg_score

def print_best_parameters(results: List[Tuple[List[int], int, float, float]]):
    if not results:
        logging.warning("No results to display.")
        return

    # Sort results by the last element (score) in descending order
    results.sort(key=lambda x: x[-1], reverse=True)

    best_result = results[0]
    hiddens, frequency, learning_rate, score = best_result

    logging.info(
        f"Best parameters:\n"
        f"  - Score: {score:.4f}\n"
        f"  - Hiddens: {', '.join(map(str, hiddens))}\n"
        f"  - Frequency: {frequency:.4f}\n"
        f"  - Learning rate: {learning_rate:.10f}"
    )

def get_os_name() -> str:
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"
    
    return "unknown"
        
        
def check_gpu():
    os_name = get_os_name()
    device_name = th.cuda.get_device_name(0)
     
    if device_name == "windows" and th.cuda.is_available():
        device_name = th.cuda.get_device_name(0)
        print(f"Operating System: {os_name}\nGPU Available: Yes\nGPU Device: {device_name}")
    else:
        print(f"Operating System: {os_name}\nGPU Available: No\nNote: CUDA is not supported on this system or the necessary drivers are not installed.")
 
def main():
    
    check_gpu()
    app_name = os.path.basename(__file__)
    
    output_folder = DirectoryManager.get_outputs_dir(app_name=app_name)
    models_dir = DirectoryManager.get_models_dir(app_name=app_name)
    logs_dir = DirectoryManager.get_logs_dir(app_name=app_name)
    
    n_timesteps = 1_000_000
    study_name = "no_physics"

    study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name=study_name)
    study.optimize(lambda trial: objective(trial, output_folder, n_timesteps, study_name, models_dir=models_dir, logs_dir=logs_dir), n_trials=100)

    results = []
    for trial in study.trials:
        hiddens = [trial.params[f'hiddens_{i}'] for i in range(trial.params['num_hiddens'])]
        results.append((hiddens, trial.params['frequency'], trial.params['learning_rate'], trial.value))

    print_best_parameters(results)

if __name__ == "__main__":
    main()
