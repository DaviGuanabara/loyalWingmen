import os
import sys

sys.path.append("..")
from modules.environments.demo_env import DemoEnvironment

steps = 10
for k in range(steps):
    x = k / steps
    reward = DemoEnvironment.linear_decay_function(x=x, min_value=0, max_value=100)
    print(x, reward)