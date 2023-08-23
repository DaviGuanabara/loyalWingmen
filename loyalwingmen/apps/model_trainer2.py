import os
import sys
sys.path.append("..")
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


from modules.environments.drone_chase_env import DroneChaseEnv
from modules.environments.randomized_drone_chase_env import RandomizedDroneChaseEnv

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

def main():
    models_dir = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs"
    logs_dir = models_dir

    suggestions = {}

    suggestions[f"hidden_{1}"] = 2048
    suggestions[f"hidden_{2}"] = 1024
    suggestions[f"hidden_{3}"] = 256
    suggestions[f"hidden_{4}"] = 2048

    suggestions["rl_frequency"] = 2
    suggestions["learning_rate"] = 1e-6
    suggestions["speed_amplification"] = 3

    suggestions["model"] = "PPO"

    hiddens = []
    for i in range(1, len(suggestions) + 1):
        key = f"hidden_{i}"
        if key in suggestions:
            hiddens.append(suggestions[key])
        else:
            break
        
        
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(environment=DroneChaseEnv, env_kwargs=suggestions)#, n_envs=1)
    specific_model_folder = ReinforcementLearningPipeline.gen_specific_folder_path(hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], dir=models_dir)
    specific_log_folder = ReinforcementLearningPipeline.gen_specific_folder_path(hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], dir=logs_dir)

    callback_list = ReinforcementLearningPipeline.create_callback_list(vectorized_environment, model_dir=specific_model_folder, log_dir=specific_log_folder, callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR], n_eval_episodes=20, debug=True)
    policy_kwargs = ReinforcementLearningPipeline.create_policy_kwargs(hiddens)
    model = ReinforcementLearningPipeline.create_model(model_type=suggestions["model"], vectorized_enviroment=vectorized_environment, policy_kwargs=policy_kwargs, learning_rate=suggestions["learning_rate"], logs_dir=specific_log_folder, debug=True)

    logging.info(model.policy)
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps=1_000_000)

    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(model, vectorized_environment, n_eval_episodes=20)
    ReinforcementLearningPipeline.save_model(model, hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], avg_reward, std_dev, models_dir)
    
    
if __name__ == '__main__':
    main()    