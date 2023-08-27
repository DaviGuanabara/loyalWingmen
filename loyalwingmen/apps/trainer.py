import time
import os
import sys
sys.path.append("..")
from stable_baselines3 import PPO, SAC
from modules.factories.callback_factory import callbacklist
from modules.utils.Logger import Logger
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from modules.environments.drone_chase_level1 import DroneChaseEnvLevel1
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    ProgressBarCallback,
    BaseCallback
)
        

def main():
    
    
    env = DroneChaseEnvLevel1(GUI=False, rl_frequency=1, debug=False)
    check_env(env, warn=True, skip_render_check=True)
    
    
    n_envs = os.cpu_count() or 1
    env_fns = [lambda: DroneChaseEnvLevel1(GUI=False, rl_frequency=1, debug=False) for i, _ in enumerate(range(n_envs))]
        
    vectorized_environment = SubprocVecEnv(env_fns)# type: ignore
        
    
    observation, info = env.reset()


    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[256, 256, 256]#dict(pi=[128, 128, 128], qf=[128, 128, 128])
                     )
    
    model = SAC(
        "MlpPolicy", 
        vectorized_environment, 
        policy_kwargs=policy_kwargs, 
        train_freq=1, 
        gradient_steps=2, 
        verbose=0, 
        learning_rate=1e-8,
        )
    
    progressbar_callback = ProgressBarCallback()
    model.learn(total_timesteps=1_000_000, callback=progressbar_callback)
    
    model.save("./sac_drone_chase_level1")


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!