# MHAST: Morphology-guided Hierarchical cell type reAssignment of Spatial Transcriptomics
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/demo-mouse_brain-brightgreen)](https://tissuumaps.scilifelab.se/brain_mouse.tmap?path=private/midl)

This repository contains the method and demos for the paper **Learned morphological features guide cell type assignment of deconvolved spatial transcriptomics**.

![](https://github.com/eduardchelebian/mhast/blob/main/midl_poster.jpg)


## Installation

We recommend creating a conda environment for running and testing the method pipeline:
```shell
conda env create -n celltyping_env -f environment.yml
```

To activate the environment:
```shell
conda activate celltyping_env
```

## Method

The function containing the hierarchical permutation method is found in `celltype_permutation.py`. To run it, simply pass:

```
A: one-hot encoded matrix (N cells x M spots) indicating the belonging of each cell to a spot
B: matrix (N cells x K features) indicating morphological features per cell
X_perm: one-hot encoded matrix (N cells x L types) indicating the initial assigned cell type per cell
```

to `X_global = hierarchical_permutations(A, X_perm, B)`, where `X_global` will be the rearranged cell types.


## Demos

* `simulated_data.ipynb` shows how the simulated Visium data was generated
* `synthetic_data.ipynb` shows how Visium data was synthesized from Xenium data
* `real_data.ipynb` shows a real use case using the Tangram cell type deconvolution method `run_tangram.py`

## Reference
Chelebian, E., Avenel, C., Leon, J., Hon, C. C., & Wahlby, C. Learned morphological features guide cell type assignment of deconvolved spatial transcriptomics. In Medical Imaging with Deep Learning. [https://openreview.net/forum?id=QfYXJUmIit](https://openreview.net/forum?id=QfYXJUmIit)
```
@inproceedings{chelebian2024learned,
  title={Learned morphological features guide cell type assignment of deconvolved spatial transcriptomics},
  author={Chelebian, Eduard and Avenel, Christophe and Leon, Julio and Hon, Chung-Chau and Wahlby, Carolina},
  booktitle={Medical Imaging with Deep Learning, 2024, Paris, France},
  year={2024},
}
```
