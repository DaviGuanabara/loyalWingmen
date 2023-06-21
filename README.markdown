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

      $ torch.cuda.get_device_name(0)
      >>>'GeForce GTX 950M'

3. install git
   - set git on path

4. download and install Visual Studio Community c/c++ IDE + Compiler (https://visualstudio.microsoft.com/vs/features/cplusplus/)
    - make sure that all tools for c++ compiler is installed. if not done properly, “wheel build” will be a problem in step 8.
5. update pip
6. update wheels
7. clone loyalwingmen (https://github.com/DaviGuanabara/loyalWingmen)
8. go to loyalWingmen\loyalwingmen folder
9. install requirements: "$ pip install -r requirements.txt"

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
   - $ python3.8 -m venv loyalwingmen
   - $ source loyalwingmen/bin/activate
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

To unsure maintanance, Gymnasium was adopted.

#### - Multiple Python Versions:
https://martinfritz.medium.com/work-with-multiple-versions-of-python-on-windows-10-eed1e5f52f07

