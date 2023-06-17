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

- [Vietnam Motorbike License Plate Public Dataset](https://bit.ly/2QzjLiC)
- [Vietnam Motorbike License Plate Public Dataset 2](https://bit.ly/3hGqvqQ)
- [Vietnam Car License Plate Public Dataset](http://j.gs/GSB1)
- [Vietnam Car License Plate Public Dataset 2](http://j.gs/GSB2)
- [Vietnam Street License Plate](http://j.gs/GSB3)

3. Manual labeling images with `LabelImg` tool

## Downloading tool

```bash
conda install pyqt=5
conda install -c anaconda lxml
git clone https://github.com/heartexlabs/labelImg.git
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
# modify the classes names in data/predefined_classes.txt [OPTIONAL]
# Change save format to YOLO [OPTIONAL]
python labelImg.py # starting the tool
```

## Tool's shortcuts

| Ctrl+u | Load all images in directory
| Ctrl+r | Change the default annotation target dir
| Ctrl+s | Save
| Ctrl+d | Copy the current label and rect box
| Ctrl+Shift+d | Delete the current image
| w | Create a rect box
| d | Next image
| a | Previous image
| del | Delete the selected rect box
| Ctrl+`+` | Zoom in
| Ctrl+`-` | Zoom out

## Labeling requirements
