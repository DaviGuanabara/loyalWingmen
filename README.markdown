# Loyal Wingmen Simulation

## Introduction
Authors:
Davi Guanabara;
Andrey Labanca.

Objective:
Simulate a Cooperative Threat Engagement by Heterogeneous Drone Swarm (CTEDS) environment inspired in https://github.com/tnferreira/cteds-high-level-decision
Solve the problem with Reinforcement Learning.

This project is based in pybullet drones (https://github.com/utiasDSL/gym-pybullet-drones)
Docs of pybullet can be found here: https://github.com/bulletphysics/bullet3/tree/master/docs


## Installation
### Windows



1. Install python3.8 (https://www.python.org/downloads/)
   - set python path

2. install latest pytorch with cuda (https://pytorch.org/)
    - make sure you installed it with cuda. to test it, try: (https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)

      $ import torch

      $ torch.cuda.is_available()
      >>>True

      $ torch.cuda.device_count()
      >>>1

      $ torch.cuda.current_device()
      >>>0

      $ torch.cuda.device(0)
      >>> <torch.cuda.device at 0x7efce0b03be0>

      $ torch.cuda.get_device_name(0) #In My Case:
      >>>'GeForce GTX 950M'

3. install git (set git on path)
   

5. download and install Visual Studio Community c/c++ IDE + Compiler (https://visualstudio.microsoft.com/vs/features/cplusplus/)
    - make sure that all tools for c++ compiler is installed. if not done properly, “wheel build” will be a problem in step 8.
    - ( Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/)
6. update pip
7. update wheels
8. Optional, Make Virtual Envorioment:
   - py -3.8 -m loyalwingmen_venv
   - ./loyalwingmen_venv/script/activate
9. clone loyalwingmen (https://github.com/DaviGuanabara/loyalWingmen)
10. go to loyalWingmen\loyalwingmen folder
11. install requirements: "$ pip install -r requirements.txt"

### MacOS
1. install xcode
2. install brew 
   - use $ unset HOMEBREW_BREW_GIT_REMOTE HOMEBREW_CORE_GIT_REMOTE
   - then: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   - (https://brew.sh/index_pt-br)
   - make sure it is in your PATH    
4. install python 3.8 with brew ($ brew install python@3.8)
   - make sure pip is in yout PATH
   - update pip ('/Applications/Xcode.app/Contents/Developer/usr/bin/python3 -m pip install --upgrade pip')

5. Optional: prepare a virtual environment
   - $ python3.8 -m venv loyalwingmen_venv
   - $ source loyalwingmen_venv/bin/activate
      
8. clone this repository
9. go to loyalwingmen/loyalwingmen
10. install requirements
   - $ pip install -r requirements.txt



### Common installation issues

#### - Not building Wheels
wheels is not building due to lack of c/c++ interpreter. On Windows, this can be solved installing a pack for c++ desktop development in Visual Studio Community, as shown in step 4 in Windows Installation
Other reason is related with python version. versions like python@3.11 brings this problem. please, use python 3.8

#### - Callback problem
Callbacks are installd with stable-baselines3[extra] (step 8)



#### - "modules" module not recognized
this error normally occurs when you try to execute any .py file inside apps folder. 

the _app.py files need to access files in parent folder. I am not sure why this error happens, but the solution
is to set sys with parents folder:
(https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)

import sys
sys.path.append('..')


I am not sure why this error message is appearing.


#### - MacOS: trace trap or Trace/BPT trap: 5
It is related with incompatibility of pynput and latest MacOS. I could not workaround it.
Please, see: https://pynput.readthedocs.io/en/latest/limitations.html


### Commom questions:
#### - Why using Gymnasium
Quoting:
"Gym did a lot of things very well, but OpenAI didn’t devote substantial resources to it beyond its initial release. The maintenance of Gym gradually decreased until Gym became wholly unmaintained in late 2020. In early 2021, OpenAI gave us control over the Gym repository."
https://farama.org/Announcing-The-Farama-Foundation

To ensure maintenance, Gymnasium was adopted.

#### - Multiple Python Versions:
https://martinfritz.medium.com/work-with-multiple-versions-of-python-on-windows-10-eed1e5f52f07

### - Invalid requirement: '_curses'
install curses:
Windows
$ pip install windows-curses

# Quick Start
The module LoyalWingmen, inside the project LoyalWingmen, is made of folders, apps, assets, and modules.
The apps folder aims to hold the applications, like training, testing, and optimization. The Demo_app.py is a demonstration file that shows a simple execution of a demo environment: two drones; the user can control one, and the other is static. 

The demo_app.py shows the basic execution of the demo environment, which is composed of two drones. A user controls one, while the other remains static as a target. The episode ends when the user reaches the target. It utilizes a keyboard listener (KeyboardListener) to enable interaction with the environment through keyboard inputs.


## Demo_app.py

The demo_app is shown below in chunks.


The code begins by importing the required modules (os, sys, DemoEnvironment, KeyboardListener).

```python
import os
import sys

sys.path.append("..")
from modules.environments.demo_env import DemoEnvironment
from modules.utils.keyboard_listener import KeyboardListener
```

Due to pynput (on KeyboardListener) incompatibility, demo_app.py is not able to run on macos properly. If you are using MacOS, the KeyboardListener will be deactivate.

```python


# ===============================================================================
# Veritifation
# ===============================================================================

MACOS = "posix"

if os.name == MACOS:
    print(os.name)
    print(
        "Demo_app.py is unable to run properly on MacOS due to pynput (on KeyboardListener) incompatibility"
    )

```

In the Setup chunk, the demo environment (DemoEnvironment) with the GUI option enabled is set. It then initializes the environment by calling the reset() function to obtain the initial observation and environment information. The keyboard listener (KeyboardListener) is initialized unless the operating system is macOS, in which case it is set to None.

```python
# ===============================================================================
# Setup
# ===============================================================================

env:DemoEnvironment = DemoEnvironment(GUI=True)
observation, info = env.reset()
keyboard_listener = KeyboardListener() if os.name != MACOS else None

```

Following that, it enters the main execution loop with 50,000 iterations. In each iteration, it checks if the KeyboardListener is set and, if so, retrieves an action with a specified intensity.
The action refers to a velocity vector, which is composed of the vector's direction, spherical coordinate angles (theta and phi), and intensity. 

the spherical coordinate system used is based on physics convention:
1. radial distance r: slant distance to the origin
2. polar angle θ (theta): angle with respect to the positive polar axis
3. azimuthal angle φ (phi): angle of rotation from the initial meridian plane

As this is a recent update, keyboard_listener still returns directions in a cartesian format. Further updates should solve this.
The spherical coordinate was chosen due to unitary constraint, which keeps the velocity vector unitary.

The 'action' is then passed to the environment's step() function, which returns the new observation, reward, termination status, and other relevant information. The code also includes a comment about the show_lidar_log() function, which has been removed due to some limitations in its functionality.
At the end of each iteration, it checks if the episode has terminated and prints a message if it has. The reset() function of the environment has been commented out to allow for a longer test without resetting the environment at each episode, giving more time to test and check.

```python
# ===============================================================================
# Execution
# ===============================================================================
for steps in range(50_000):
    action = (
        keyboard_listener.get_action(intensity=0.005)
        if keyboard_listener is not None
        else [math.pi/2, 0, 0.1]
    )
    observation, reward, terminated, truncated, info = env.step(action)
    #env.show_lidar_log()

    if terminated:
        print("Episode terminated")

        # I preferred to remove the reset to be able to make a long test
        # env.reset()

```

That is the simplest code needed to execute an evironment.

