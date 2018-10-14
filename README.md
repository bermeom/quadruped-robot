# Quadruped-robot
### Requirements
The project has some dependencies:
- Install gym dependencies(more information [gym repository](https://github.com/openai/gym))  
-- On Ubuntu 16.04:
```sh
$   apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
        cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg
```
-- On Ubuntu 18.04:
```sh
$   apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb
```    
- Install Python 3.6 
--  On Ubuntu 16.04
```sh
$   sudo add-apt-repository ppa:jonathonf/python-3.6
$   sudo apt-get update
$   sudo apt-get install -y python3.6 python3.6-dev 
$   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
$   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
$   sudo update-alternatives --config python3
$   python3 -V
```
- Install [MuJoCo](http://www.mujoco.org/)
1. Obtain a 30-day free trial on the [MuJoCo](http://www.mujoco.org/) website or free license if you are a student. The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 150 binaries for Linux, OSX, or Windows.
3. Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (the `mjkey.txt` file from your email) at `~/.mujoco/mjkey.txt`.
4. Add the following lines in `.bashrc` at the end:
```sh
export MUJOCO_PATH="$HOME/.mujoco/mjpro150"
export MUJOCO_LICENSE_PATH="${MUJOCO_PATH}/bin/mjkey.txt"
export PATH="${PATH}:${MUJOCO_PATH}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_PATH}/bin"
```
5. Test if MuJoCo is working
```sh
$   cd ~/.mujoco/mjpro150/bin
$   ./simulate ../model/humanoid.xml 
```
Note: if you have problems with GLFW or something later, you should add these lines in `.bashrc` at the end:
```sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia-396"
export LD_PRELOAD="$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so"
```
- Install [GYM](https://github.com/openai/gym) and [mujoco-py](https://github.com/openai/mujoco-py): 
```sh
$   sudo -H pip3 install -U gym
$   sudo -H pip3 install -U '^Cjoco-py<1.50.2,>=1.50.1'
```
Test if the installation was successful, run this code with `python3`:
```python
import gym
import time
env = gym.make('HalfCheetah-v2')
env.reset()
env.render()
time.sleep( 5 )
```
