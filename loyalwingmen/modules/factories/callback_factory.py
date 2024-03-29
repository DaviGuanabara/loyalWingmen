from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class StorageForEvalCallback:
    best_mean_reward: float = -math.inf


class StoreDataOnBestCallback(BaseCallback):
    def __init__(self, storage: StorageForEvalCallback = None, verbose: int = 0):
        assert (
            storage is not None
        ), "`StoreDataOnBestCallback` callback must be initialized with an `StorageForCallback`"

        super(StoreDataOnBestCallback, self).__init__(verbose=verbose)
        self.best_mean_reward = -math.inf
        self.storage = storage

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        assert self.parent is not None, (
            "`StoreDataOnBestCallback` callback must be used " "with an `EvalCallback`"
        )

        # self.best_mean_reward = self.parent.best_mean_reward
        self.storage.best_mean_reward = self.parent.best_mean_reward

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


def gen_eval_callback(env, log_path, model_path, eval_freq=1000, storage=None):
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


def gen_checkpoint_callback(save_freq, save_path, n_envs):
    save_freq = max(save_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)

    return checkpoint_callback


def callbacklist(
    env, log_path="./logs/", model_path="./models/", n_envs=1, save_freq=10000
):
    list = []

    # trigger its child callback when there is a new best model
    storageForEvalCallback = StorageForEvalCallback()
    eval_callback = gen_eval_callback(
        env, log_path, model_path, eval_freq=save_freq, storage=storageForEvalCallback
    )
    checkpoint_callback = gen_checkpoint_callback(save_freq, model_path, n_envs)
    progressbar_callback = ProgressBarCallback()

    list.append(eval_callback)
    list.append(checkpoint_callback)
    list.append(progressbar_callback)

    return CallbackList(list), storageForEvalCallback
