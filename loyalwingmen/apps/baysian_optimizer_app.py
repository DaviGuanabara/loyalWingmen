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

    

def rl_pipeline(suggested_parameters: Tuple[int, int, int, int, float], n_timesteps: int, models_dir: str, logs_dir: str, n_eval_episodes: int = 100) -> Tuple[float, float, float]:
    
    hidden_1, hidden_2, hidden_3, frequency, learning_rate = suggested_parameters
    hiddens = list((hidden_1, hidden_2, hidden_3))
    
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(frequency)
    specific_model_folder = ReinforcementLearningPipeline.gen_specific_model_folder_path(hiddens, frequency, learning_rate, models_dir=models_dir)
    callback_list = ReinforcementLearningPipeline.create_callback_list(vectorized_environment, model_dir=specific_model_folder, log_dir=logs_dir, callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR], n_eval_episodes=n_eval_episodes, debug=True)
    policy_kwargs = ReinforcementLearningPipeline.create_policy_kwargs(hiddens)
    model = ReinforcementLearningPipeline.create_ppo_model(vectorized_environment, policy_kwargs, learning_rate, debug=True)

    logging.info(model.policy)
    #TODO: in train model, the callback list EVAL is saving the best_model on the folder models_dir, but it should be in
    # the folder of ReinforcementLearningPipeline.save_model. So, the operation is being done twice and the best model if getting lost.
    #VERIFY IF IT WERE SOLVED WITH ReinforcementLearningPipeline.gen_specific_model_folder_path.
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps)
    
    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(model, vectorized_environment, n_eval_episodes=n_eval_episodes)
    ReinforcementLearningPipeline.save_model(model, hiddens, frequency, learning_rate, avg_reward, std_dev, models_dir)
    
    return avg_reward, std_dev, num_episodes

def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate


def suggest_parameters(trial: Trial) -> Tuple[int, int, int, int, float]:
    #num_hiddens = trial.suggest_int('num_hiddens', 3, 4)
    #num_hiddens = trial.suggest_int('num_hiddens', 3, 3)
    #num_hiddens = 3
    #hiddens = [trial.suggest_int(f'hiddens_{i}', 1, 4) * 128 for i in range(num_hiddens)]
    #hiddens = [trial.suggest_categorical(f'hiddens_{i}', [128, 256, 512]) for i in range(num_hiddens)]
    
    #hidden_1 = trial.suggest_categorical(f'hiddens_1', [128, 256, 512])
    #hidden_2 = trial.suggest_categorical(f'hiddens_1', [128, 256, 512])
    #hidden_3 = trial.suggest_categorical(f'hiddens_1', [128, 256, 512])
    
    hidden_1 = 128
    hidden_2 = 128
    hidden_3 = 128

    #frequency = trial.suggest_int('frequency', 1, 2) * 15
    #frequency = trial.suggest_categorical('frequency', [15, 30])
    frequency = 1
    exponent = trial.suggest_int('exponent', -9, -7)
    learning_rate = 10 ** exponent
    
    logging.info(
        f"Suggested Parameters:\n"
        f"  - Hiddens: {', '.join(map(str, [hidden_1, hidden_2, hidden_3]))}\n"
        f"  - Frequency: {frequency}\n"
        f"  - Learning Rate: {learning_rate:.2e}"  # Change the precision value as needed
    )

    
    
    return hidden_1, hidden_2, hidden_3, frequency, learning_rate


def objective(trial: Trial, output_folder: str, n_timesteps: int, study_name: str, models_dir: str, logs_dir:str) -> float:
    
    #hidden_1, hidden_2, hidden_3, frequency, learning_rate = suggest_parameters(trial)
    suggested_parameters = suggest_parameters(trial)
    
    avg_score, std_deviation, n_episodes = rl_pipeline(suggested_parameters, n_timesteps=n_timesteps, models_dir=models_dir, logs_dir=logs_dir)
    logging.info(f"Avg score: {avg_score}")

    print("saving results...")
    result = list(suggested_parameters)
    result.append(avg_score)
    result.append(std_deviation)
    headers = ["hidden_1", "hidden_2", "hidden_3", 'frequency', 'learning_rate', 'value', 'std_deviation']
    
    try:
        ReinforcementLearningPipeline.save_results_to_excel(output_folder, f"results_{study_name}.xlsx", result, headers)
    except:
        ReinforcementLearningPipeline.save_results_to_excel(output_folder, f"results_{study_name}_1.xlsx", result, headers)        
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
    
    if os_name == "windows" and th.cuda.is_available():
        device_name = th.cuda.get_device_name(0)
        logging.info(f"Operating System: {os_name}\nGPU Available: Yes\nGPU Device: {device_name}")
    else:
        logging.info(f"Operating System: {os_name}\nGPU Available: No\n Note: CUDA is not supported on this system or the necessary drivers are not installed.")
        if os_name == "windows":
            raise Exception("Fix CUDA support on Windows")
        
 
def main():
    
    check_gpu()
    n_timesteps = 1_000_000
    n_timesteps_in_millions = int(n_timesteps / 1e6)
    study_name = f"no_physics_in_{n_timesteps_in_millions}M_steps_reward_distance_low_frequency_speed_amplification"
    app_name = os.path.basename(__file__)
    app_name = os.path.join(app_name, study_name)
    
    output_folder = DirectoryManager.get_outputs_dir(app_name=app_name)
    models_dir = DirectoryManager.get_models_dir(app_name=app_name)
    logs_dir = DirectoryManager.get_logs_dir(app_name=app_name)
    
    
    
    
    #vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(n_envs, frequency)

    
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name=study_name)
    study.optimize(lambda trial: objective(trial, output_folder, n_timesteps, study_name, models_dir=models_dir, logs_dir=logs_dir), n_trials=100)

    results = []
    for trial in study.trials:
        hiddens = [trial.params[f'hiddens_{i}'] for i in range(trial.params['num_hiddens'])]
        results.append((hiddens, trial.params['frequency'], trial.params['learning_rate'], trial.value))

    print_best_parameters(results)

if __name__ == "__main__":
    main()
