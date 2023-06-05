## Installation
### Windows







1. Install latest python (https://www.python.org/downloads/)
   - set python path

2. install latest pytorch with cuda (https://pytorch.org/)
    - make sure you installed it with cuda. to test it, try:
(https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)
>>> import torch

>>> torch.cuda.is_available()
True

>>> torch.cuda.device_count()
1

>>> torch.cuda.current_device()
0

>>> torch.cuda.device(0)
<torch.cuda.device at 0x7efce0b03be0>

>>> torch.cuda.get_device_name(0)
'GeForce GTX 950M'

3. install git
   - set git on path

4. download and install Visual Studio Community c/c++ IDE + Compiler (https://visualstudio.microsoft.com/vs/features/cplusplus/)
    - make sure that all tools for c++ compiler is installed. if not done properly, “wheel build” will be a problem in step 8.
5. update pip
6. update wheels
7. install gymnasium
8. install stable-baselines3[extra] (need for callbacks)
9. install stable baselines 3 bleeding-edge version, required to use gymnasium (https://stable-baselines3.readthedocs.io/en/master/guide/install.html).
10. clone loyalwingmen (https://github.com/DaviGuanabara/loyalWingmen)
11. go to loyalWingmen\loyalwingmen folder
12. install loyalwingmen module: "$ pip install ."
13. install requirements: "$ pip install -r requirements.txt"

### Common installation issues

#### Not building Wheels
wheels is not building due to lack of c/c++ interpreter. On Windows, this can be solved installing a pack for c++ desktop development in Visual Studio Community, as shown in step 4 in Windows Installation


#### Callback problem
Callbacks are installd with stable-baselines3[extra] (step 8)

#### Why using Gymnasium
Quoting:
"Gym did a lot of things very well, but OpenAI didn’t devote substantial resources to it beyond its initial release. The maintenance of Gym gradually decreased until Gym became wholly unmaintained in late 2020. In early 2021, OpenAI gave us control over the Gym repository."
https://farama.org/Announcing-The-Farama-Foundation

To unsure maintanance, Gymnasium was adopted.

#### "modules" module not recognized
this error normally occurs when you try to execute any .py file inside apps folder. 

the _app.py files need to access files in parent folder. I am not sure why this error happens, but the solution
is to set sys with parents folder:
(https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)

import sys
sys.path.append('..')


I am not sure why this error message is appearing.


