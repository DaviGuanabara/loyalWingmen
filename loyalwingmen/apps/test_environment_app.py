from modules.environments.loyalwingmen.lidar_env2 import DroneLidar2
from modules.utils.keyboard_listener import KeyboardListener
import sys


sys.path.append("..")

# from modules.utils.keyboard_listener import KeyboardListener


env = DroneLidar2(GUI=True)
action = [1, 0, 0, 0.01]

# funciona para sb3 a partir de 2.0.0
observation, info = env.reset()
keyboard_listener = KeyboardListener()
for steps in range(50_000):
    action = keyboard_listener.get_action(intensity=0.005)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)
    # TODO: display text e logreturn pode ser incorporado pelo ambiente.
    env.show_lidar_log()

    # log_returns(observation, reward, action)
    # if terminated:
    # observation, info = env.reset()
