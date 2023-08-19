import os
import sys
sys.path.append("..")
import logging

from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from modules.environments.drone_chase_env import DroneChaseEnv
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import callbacklist, CallbackType
from typing import List, Tuple
from datetime import datetime

from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple
import re

class DirectoryManager:

    @staticmethod
    def find_base_dir(current_path: str, target_dir: str = "loyalwingmen", debug: bool = True) -> str:
        if debug:
            logging.basicConfig(level=logging.DEBUG)

        iterations = 0
        max_iterations = 5

        while iterations < max_iterations:
            current_path, base_dir = os.path.split(current_path)

            logging.debug(f"Current path: {current_path}")
            logging.debug(f"Base dir: {base_dir}")

            if base_dir == target_dir:
                path = os.path.join(current_path, base_dir)
                logging.debug(f"path: {os.path.join(current_path, base_dir)}")
                return path

            has_reached_root = True if not current_path or current_path == os.path.sep else False
            has_reached_max_interations = iterations == max_iterations - 1
            if has_reached_root or has_reached_max_interations:
                script_directory = os.path.dirname(os.path.abspath(__file__))
                return script_directory

            iterations += 1

        # This line should not be reached, but it's here as a safety measure
        logging.warning("Unexpected termination of loop.")
        return ""



    @staticmethod
    def get_base_dir(debug: bool = False) -> str:
        path = os.path.abspath(__file__)
        base_dir = DirectoryManager.find_base_dir(path)
        logging.error("Could not find base dir") if not base_dir and debug else None
        
        return base_dir

    @staticmethod
    def get_outputs_dir(app_name: str, debug: bool = False) -> str:
        base_dir = DirectoryManager.get_base_dir()
        outputs_dir = os.path.join(base_dir, "outputs", app_name)
        logging.debug(f"(get_outputs_dir) outputs_dir: {outputs_dir}") if debug else None

        DirectoryManager.create_directory(outputs_dir)
              
        return outputs_dir

    @staticmethod
    def get_logs_dir(app_name: str, debug: bool = False) -> str:
        outputs_dir = DirectoryManager.get_outputs_dir(app_name, debug)
        logs_dir = os.path.join(outputs_dir, "logs")
        DirectoryManager.create_directory(logs_dir) 
        return logs_dir

    @staticmethod
    def get_models_dir(app_name: str, debug: bool = False) -> str:
        outputs_dir = DirectoryManager.get_outputs_dir(app_name, debug)
        models_dir = os.path.join(outputs_dir, "models")
        DirectoryManager.create_directory(models_dir) 
        return models_dir
    
    @staticmethod
    def create_directory(directory_path: str) -> str:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path
    
