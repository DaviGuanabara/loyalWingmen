from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


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


def gen_eval_callback(env, log_path, model_path, eval_freq=1000):
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3, min_evals=5, verbose=0
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback,
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
    eval_callback = gen_eval_callback(env, log_path, model_path, eval_freq=save_freq)
    checkpoint_callback = gen_checkpoint_callback(save_freq, model_path, n_envs)
    progressbar_callback = ProgressBarCallback()

    list.append(eval_callback)
    list.append(checkpoint_callback)
    list.append(progressbar_callback)

    return CallbackList(list)
