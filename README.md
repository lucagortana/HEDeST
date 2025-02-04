# Plugin tool

![figure](references/method.png)

## Installation of the requirements
To create your conda environment and install the requirements with ``pip``
```
$ conda create -y -n name-env python=3.9
$ conda activate name-env
$ pip install -r requirements.txt
```
To install pytorch and tensorboard
```
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install tensorboard
```
To install openslide
```
$ conda install -c conda-forge openslide=3.4.1
```

## Some classic errors
During segmentation, you can get the ``OSError: [Errno 39] Directory not empty: 'cache'`` error. Make sure to delete everything you have in this repository and apply chmod 777.
