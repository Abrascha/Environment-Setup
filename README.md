# Environment-Setup
Environment-Setup for Mashine Learning / Deep Learning

Seting up different CUDA and Pytorch Version on one Ubuntu at the same time

Keyword: Ubuntu 18.04, Ubuntu 16.04, CUDA, Pytorch, Anaconda, Python

---
<span id="jump"></span>
## Check first
[CUDA Toolkit and Compatible Driver Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)

| CUDA Toolkit                                      | Linux x86_64 Driver Version | Windows x86_64 Driver Version |
| :------------------------------------------------ | :-------------------------- | :---------------------------- |
| CUDA 11.0.189 RC                                  | >= 450.36.06                | >= 451.22                     |
| CUDA 10.2.89                                      | >= 440.33                   | >= 441.22                     |
| CUDA 10.1 (10.1.105 general release, and updates) | >= 418.39                   | >= 418.96                     |
| CUDA 10.0.130                                     | >= 410.48                   | >= 411.31                     |
| CUDA 9.2 (9.2.148 Update 1)                       | >= 396.37                   | >= 398.26                     |
| CUDA 9.2 (9.2.88)                                 | >= 396.26                   | >= 397.44                     |
| CUDA 9.1 (9.1.85)                                 | >= 390.46                   | >= 391.29                     |
| CUDA 9.0 (9.0.76)                                 | >= 384.81                   | >= 385.54                     |
| CUDA 8.0 (8.0.61 GA2)                             | >= 375.26                   | >= 376.51                     |
| CUDA 8.0 (8.0.44)                                 | >= 367.48                   | >= 369.30                     |
| CUDA 7.5 (7.5.16)                                 | >= 352.31                   | >= 353.66                     |
| CUDA 7.0 (7.0.28)                                 | >= 346.46                   | >= 347.62                     |


The Archived Releases of CUDA support Ubuntu Version like:    
- Ubuntu 18.04: >= CUDA 10.0
- Ubuntu 16.04: >= CUDA 8.0

So if you want to use the CUDA Version 8.0 - 9.2, just choose the Ubuntu 16.04 Version. 

The Version of [Pytorch](https://pytorch.org/get-started/locally/) depends on CUDA Version

## Ubuntu Installation
[Downloads Link](https://releases.ubuntu.com/?_ga=2.26883769.549160946.1592251480-1982882452.1590480922)

[Install Ubuntu desktop](https://ubuntu.com/tutorials/tutorial-install-ubuntu-desktop)

[Install Ubuntu Server](https://ubuntu.com/tutorials/tutorial-install-ubuntu-server#1-overview)

## Pre-Installation
``` bash
sudo apt update
sudo apt install gcc
```    
Disable nouveau:
``` bash
# check nouveau
lsmod | grep nouveau
# Create a file at /etc/modprobe.d/blacklist-nouveau.conf
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
# copy with the following contents:
blacklist nouveau
options nouveau modeset=0
# save and exit
# Regenerate the kernel initramfs:
sudo update-initramfs -u
# check it again
lsmod | grep nouveau
```

## Nvidia Driver

``` bash
# check the nvidia Graphic
lspci | grep -i nvidia
# remove old Nvidai-driver
sudo apt purge nvidia*
# add repository and update 
sudo add-apt-repository ppa:graphics-drivers
sudo apt update
# install ubuntu-drivers package
sudo apt install ubuntu-drivers-common
# check available driver
ubuntu-drivers devices
# install the recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
# NVIDIA System Management Interface: to check availability
nvidia-smi
```

## CUDA
[Archived Releases](https://developer.nvidia.com/cuda-toolkit-archive)

The Version of CUDA depends on Ubuntu's Version and Nvidia-Driver's Version. [Check first](#jump)

Example: Ubuntu 16.04 + CUDA 10.2
``` bash
# Recommanded: Download the runfile(local)
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run cuda_10.2.89_440.33.01_linux.run
# add executable rights
sudo chmod +x cuda_10.2.89_440.33.01_linux.run
# run it
sudo ./cuda_10.2.89_440.33.01_linux.run
# !!! Important !!!
# 1. Do not choose Installation of Graphics Driver! You did it before.
# 2. During the Installation. It need to change the Installation path for different Version
# Options -> Library install path (Blank for system default)
# /usr/local/cuda-10.2
```

For CUDA version-choose, I set aliases for Terminal:
```
vim ~/.bash_aliases 
```
Add the line in ~/.bash_aliases
``` bash
# change "cuda-10.2" to the installations version
alias env_cuda10.2='export PATH=/usr/local/cuda-10.2/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH'
# save and exit: Enter it
:wq
# restart the Terminal
```

``` bash
# Enter env_cuda10.2 to choose the CUDA Version 10.2
env_cuda10.2
# check the CUDA Version activated now
nvcc -V
# output: Cuda compilation tools, release 10.2, V10.2.89
```


## Anaconda
``` bash
# Download
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
# run it
./Anaconda3-2020.02-Linux-x86_64.sh
```
or
[Download Site](https://www.anaconda.com/products/individual)

Create conda enviroment for different Pytorch and Python version:
``` bash
# name and pyhton-Version depends on you
conda create --name dp-pythorch1.5 python=3.7
# activate conda enviroment
conda activate dp-pythorch1.5
# you can set different name for different pytorch and python requirements
# deactivate
conda deactivate
```


## pytorch
[Last Version](https://pytorch.org/get-started/locally/)

[Previous Version](https://pytorch.org/get-started/previous-versions/)

``` bash
#Choose the conda enviroment first
conda activate dp-pythorch1.5
# Installation pytorch 1.5 for CUDA 10.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Verification:
``` python
# save the python3 script as verification.py
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
```
Run the verification.py
``` bash
python verification.py
```
The output should like:

    tensor([[0.8614, 0.9250, 0.5427],
            [0.0077, 0.7420, 0.0814],
            [0.2043, 0.8488, 0.5750],
            [0.9281, 0.0171, 0.5761],
            [0.3821, 0.9040, 0.9519]])
    True

## Useage

1. Open Terminal
2. Activate CUDA: `env_cuda10.2`
3. Activate conda enviroment: `conda activate dp-pythorch1.5`
4. Execute your script: `python verification.py`
