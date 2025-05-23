# Plugin tool

![figure](references/method.png)

## Installation of the requirements
To create your conda environment and install the requirements with ``pip``
```
conda create -y -n name-env python=3.9
conda activate name-env
pip install -r requirements.txt
```
To install pytorch, torch-scatter and tensorboard
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu116-cp39-cp39-linux_x86_64.whl
pip install tensorboard
```
To install openslide
```
conda install -c conda-forge openslide=3.4.1
```

## Some classic errors
During segmentation, you can get the ``OSError: [Errno 39] Directory not empty: 'cache'`` error. Make sure to delete everything you have in this repository and apply chmod 777. \
Also, you can get the Openslide's error ``openslide.lowlevel.OpenSlideUnsupportedFormatError: Unsupported or missing image file``. In that case, we recommend to check the properties of your file with the command ``openslide-show-properties your_file.tif``. If your file is indeed incompatible with Openslide, then you can try :
```
vips tiffsave Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif output-pyramidal.tif --tile --pyramid --bigtiff --compression jpeg --Q 90
```
