import cProfile
import sys
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env
from loyalwingmen.modules.utils.keyboard_listener import KeyboardListener
from loyalwingmen.modules.environments.level1_environment import Level1
from loyalwingmen.modules.utils.displaytext import log
import numpy as np
import pstats

env = Level1(GUI=False, rl_frequency=30, debug=True)
check_env(env)

def on_avaluation_step(n=1_000):
    max_allowed_time = 1 / env.environment_parameters.rl_frequency  # Convert frequency to time. For 30Hz, this is about 0.0333 seconds.
    
    successful_evaluations = 0
    worst_case = -1
    best_case = 1_000_000  # initialize best case to a large value
    total_time = 0  # accumulator for total time spent
    
    for _ in range(n):
        start_time = time.time()
        env.step(np.zeros(4))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time  # accumulate the elapsed time
        
        if elapsed_time <= max_allowed_time:
            successful_evaluations += 1
        
        if worst_case < elapsed_time:
            worst_case = elapsed_time
        
        if best_case > elapsed_time:  # update best case if the current elapsed time is shorter
            best_case = elapsed_time
    
    average_frequency = n / total_time
    
    
    print(f"In {successful_evaluations} of {n} evaluations of env.step, the elapsed time was within the limit of {max_allowed_time:.4f} seconds.")
    print(f"Worst-case frequency: {1/worst_case:.4f} hz")
    if best_case == 0:
        print("Best-case frequency: Infinite (elapsed time was zero)")
    else:
        print(f"Best-case frequency: {1/best_case:.4f} hz")

    print(f"Average frequency: {average_frequency:.4f} hz")
    if successful_evaluations == n:
        print(f"Success: All {n} evaluations of env.step were within the limit of {max_allowed_time:.4f} seconds.")



cProfile.run('on_avaluation_step()', 'result.prof')
stats = pstats.Stats('result.prof')
stats.sort_stats('cumulative').print_stats(10)  # Show the top 10 functions by cumulative time
