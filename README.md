## Installation of the requirements
To create your conda environment and install the requirements with ``pip``
```
$ conda create -y -n name-env python=3.9
$ conda activate name-env
$ pip install -r requirements.txt
```
To install pytorch
```
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
To install openslide
```
$ conda install -c conda-forge openslide=3.4.1
```
