import os
from typing import Optional, Tuple
import numpy as np
import math
from dataclasses import dataclass

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    ProgressBarCallback,
    BaseCallback
)
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecEnv

from typing import List, Tuple

from enum import Enum
from typing import List
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv


class CallbackType(Enum):
    EVAL = "eval"
    CHECKPOINT = "checkpoint"
    PROGRESSBAR = "progressbar"


def callbacklist(
    env: VecEnv, 
    log_path: str = "./logs/",
    model_path: str = "./models/",
    n_envs: int = 1,
    save_freq: int = 10000,
    callbacks_to_include: List[CallbackType] = [CallbackType.EVAL, CallbackType.CHECKPOINT, CallbackType.PROGRESSBAR]
) -> CallbackList:
    list_callbacks = []

    if CallbackType.EVAL in callbacks_to_include:
        # Check if directories are writable
        assert os.access(log_path, os.W_OK), "log_path must be a writable directory"
        assert os.access(model_path, os.W_OK), "model_path must be a writable directory"

        # Check if eval_freq is valid
        assert isinstance(save_freq, int) and save_freq > 0, "save_freq must be a positive integer"

        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=0
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_path,
            log_path=log_path,
            eval_freq=save_freq,
            deterministic=True,
            render=False,
            callback_after_eval=stop_train_callback,
        )

        list_callbacks.append(eval_callback)

    if CallbackType.CHECKPOINT in callbacks_to_include:
        # Check if save_path is writable
        assert os.access(model_path, os.W_OK), "model_path must be a writable directory"

        # Check if save_freq and n_envs are valid
        assert isinstance(save_freq, int) and save_freq > 0, "save_freq must be a positive integer"
        assert isinstance(n_envs, int) and n_envs > 0, "n_envs must be a positive integer"

        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)

        list_callbacks.append(checkpoint_callback)

    if CallbackType.PROGRESSBAR in callbacks_to_include:
        progressbar_callback = ProgressBarCallback()
        list_callbacks.append(progressbar_callback)

    return CallbackList(list_callbacks)
