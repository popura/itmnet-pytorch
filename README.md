# iTMNet
This is an implementation for *iTM-Net: Deep Inverse Tone Mapping Using Novel Loss Function Considering Tone Mapping Operator*.

[Project page](https://sites.google.com/view/kinoshita-yuma-en/publications/itm-net),
[DOI](https://doi.org/10.1109/ACCESS.2019.2919296)

When you use this implementation for your research work,
please cite the paper.

The following is the bibtex entry.
```
@article{kinoshita2019itmnet,
author = {Kinoshita, Yuma and Kiya, Hitoshi},
doi = {10.1109/ACCESS.2019.2919296},
issn = {2169-3536},
journal = {IEEE Access},
volume = {7},
number = {1},
pages = {73555--73563},
title = {{iTM-Net: Deep Inverse Tone Mapping Using Novel Loss Function Considering Tone Mapping Operator}},
url = {https://ieeexplore.ieee.org/document/8723346/},
month = {May},
year = {2019}
}
```

# Requirements
- Python 3.9 or later
- Pytorch 1.8 or later
- hdrpy 0.1.0 or later (in the `external` directory)
  
  Repo: https://github.com/popura/hdrpy
- deepy 0.5.0 or later (in the `external` directory)

For other requirements, see pyproject.toml

# Getting started
1. Clone this repository
    ```
    git clone https://github.com/popura/itmnet-pytorch.git
    cd itmnet-pytorch
    ```
1. Install requirements.

    If you use poetry as a package manager, it is done by
    ```
    poetry install
    ```
1. Train iTM-Net.
   All outputs including trained models will be written in the `history` directory.
    ```
    poetry run python ./src/train.py
    ```
1. Test.
   All outputs will be written in the `result` directory.
    ```
    poetry run python ./src/test.py
    ```
