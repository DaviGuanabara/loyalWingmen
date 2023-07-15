import os
import sys

sys.path.append("..")
from modules.environments.battlefield_env import Battlefield
from modules.utils.keyboard_listener import KeyboardListener



"""
Demo.app is a file made to show a simple execution of an environment.
I hope you enjoy it.
Problem:
It is necessary to highlight that KeyboardListener() won't work in latest MacOS:

'Recent versions of macOS restrict monitoring of the keyboard for security reasons. For that reason, one of the following must be true:

1. The process must run as root.
3. Your application must be white listed under Enable access for assistive devices. 
   Note that this might require that you package your application, since otherwise the entire Python installation must be white listed.
2. On versions after Mojave, you may also need to whitelist your terminal application if running your script from a terminal.'

More informations in: https://pynput.readthedocs.io/en/latest/limitations.html

I were not able to make it work in MacOS Ventura on M1 Pro
"""

# ===============================================================================
# Veritifation
# ===============================================================================

MACOS = "posix"

if os.name == MACOS:
    print(os.name)
    print(
        "Demo_app.py is unable to run properly on MacOS due to pynput (on KeyboardListener) incompatibility"
    )

# ===============================================================================
# Setup
# ===============================================================================

env = Battlefield(GUI=True)
observation, info = env.reset()
keyboard_listener = KeyboardListener() if os.name != MACOS else None

# ===============================================================================
# Execution
# ===============================================================================
for steps in range(50_000):
    action = (
        keyboard_listener.get_action(intensity=0.005)
        if keyboard_listener is not None
        else [1, 0, 0, 0.5]
    )
    observation, reward, terminated, truncated, info = env.step(action)
    env.show_lidar_log()

    if terminated:
        print("Episode terminated")

        # I preferred to remove the reset to be able to make a long test
        # env.reset()
