# unitree A1 sim2sim for test reinforcement learning by webots

## perpare envirment for webots and pytorch
1. Download webots R2023b form [WebotsR2023](https://www.cyberbotics.com/#download)

2. Download anaconda from [anaconda](https://www.anaconda.com/download)

3. Create a new conda envirment:
`conda create -n env_name python=3.8`

4. Add webots address to python install path: Create a file named as webots.pth at python(conda) install path
(eg: my conda pthon address is: C:\Users\12466\.conda\envs\DeepReinforce). Add the webots head path to the webots.pth:
`C:\Program Files\Webots\lib\controller\python
C:\Program Files\Webots\msys64\mingw64\bin\
C:\Program Files\Webots\msys64\mingw64\bin\cpp`

5. install conda and pytorch in conda env

## Run the sim code

1. start your conda envirment:
`conda activate env_name`
2. open webots world at : project+address/worlds/unitree_a1_pytorch_simTosim.wbt
3. python run the code by
`python RL_Contorller.py`
 
