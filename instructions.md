1. Download Anaconda3 and create environment

```bash
cd $HOME
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
zsh Anaconda3-2023.03-1-Linux-x86_64.sh
```

- Follow instructions of the installer to install anaconda3

```bash
export ENV=torch
conda init zsh
conda create -n $ENV python=3.9 -y
conda activate $ENV
```

## Install libraries and dependencies

```bash
# Install pytorch with GPU supports
conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.8.0 -y # CUDA vs cuDNN
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # pytorch
pip install -r requirements.txt # utilities
```

2. Collect data from internet

[Vietnam License Plate Public Dataset](https://bit.ly/2QzjLiC)
