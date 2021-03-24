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

For other requirements, see pyproject.toml

# Getting started
1. Clone this repository
    ```
    git clone https://github.com/popura/itmnet-pytorch.git
    cd itmnet-pytorch
    ```
1. Install requirements
    ```
    pip install -r requirements.txt
    ```
1. Train iTM-Net
    ```
    python ./src/train.py
    ```
1. Test
    ```
    python ./src/test.py
    ```
