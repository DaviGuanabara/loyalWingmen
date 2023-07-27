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
from stable_baselines3.common.vec_env import VecEnv


@dataclass
class StorageForEvalCallback:
    best_mean_reward: float = -math.inf

#TODO: self.storage.best_mean_reward = self.parent.best_mean_reward
# Expression of type "tuple[CallbackList, StorageForEvalCallback | None]" cannot be assigned to return type "Tuple[CallbackList, StorageForEvalCallback]"
#  Type "StorageForEvalCallback | None" cannot be assigned to type "StorageForEvalCallback"
#    Type "None" cannot be assigned to type "StorageForEvalCallback"#

#TODO repensar no StoreData: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#callbacks-evaluate-agent-performance
class StoreDataOnBestCallback(BaseCallback):
    def __init__(self, storage: Optional[StorageForEvalCallback] = None, verbose: int = 0):
        assert storage is not None, "`StoreDataOnBestCallback` callback must be initialized with an `StorageForCallback`"

        super().__init__(verbose=verbose)
        self.storage = storage

    def _on_step(self) -> bool:
        assert self.parent is not None, "`StoreDataOnBestCallback` callback must be used with an `EvalCallback`"

        # Only update storage if a new best mean reward is found
        if self.storage.best_mean_reward != self.parent.best_mean_reward:
            self.storage.best_mean_reward = self.parent.best_mean_reward

        return True


def gen_eval_callback(
    env: VecEnv, 
    log_path: str, 
    model_path: str, 
    eval_freq: int = 1000, 
    storage: Optional[StorageForEvalCallback] = None
) -> EvalCallback:
    # Check if directories are writable
    assert os.access(log_path, os.W_OK), "log_path must be a writable directory"
    assert os.access(model_path, os.W_OK), "model_path must be a writable directory"

    # Check if eval_freq is valid
    assert isinstance(eval_freq, int) and eval_freq > 0, "eval_freq must be a positive integer"

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3, min_evals=5, verbose=0
    )

    callback_on_new_best = None
    if storage is not None:
        callback_on_new_best = StoreDataOnBestCallback(storage)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback,
        callback_on_new_best=callback_on_new_best,
    )

    return eval_callback


def gen_checkpoint_callback(
    save_freq: int, 
    save_path: str, 
    n_envs: int
) -> CheckpointCallback:
    # Check if save_path is writable
    assert os.access(save_path, os.W_OK), "save_path must be a writable directory"

    # Check if save_freq and n_envs are valid
    assert isinstance(save_freq, int) and save_freq > 0, "save_freq must be a positive integer"
    assert isinstance(n_envs, int) and n_envs > 0, "n_envs must be a positive integer"

    save_freq = max(save_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)

    return checkpoint_callback


from typing import List, Tuple

def callbacklist(
    env: VecEnv, 
    log_path: str = "./logs/", 
    model_path: str = "./models/", 
    n_envs: int = 1, 
    save_freq: int = 10000, 
    callbacks_to_include: List[str] = ["eval", "checkpoint", "progressbar"]
) -> Tuple[CallbackList, StorageForEvalCallback]:
    list_callbacks = []

    if "eval" in callbacks_to_include:
        # trigger its child callback when there is a new best model
        storageForEvalCallback = StorageForEvalCallback()
        eval_callback = gen_eval_callback(
            env, log_path, model_path, eval_freq=save_freq, storage=storageForEvalCallback
        )
        list_callbacks.append(eval_callback)
    else:
        storageForEvalCallback = None

    if "checkpoint" in callbacks_to_include:
        checkpoint_callback = gen_checkpoint_callback(save_freq, model_path, n_envs)
        list_callbacks.append(checkpoint_callback)
    
    if "progressbar" in callbacks_to_include:
        progressbar_callback = ProgressBarCallback()
        list_callbacks.append(progressbar_callback)

    return CallbackList(list_callbacks), storageForEvalCallback

