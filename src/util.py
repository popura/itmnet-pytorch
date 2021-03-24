import typing

import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary

from omegaconf import DictConfig

from itmnet import ITMNet


HDR_IMG_EXTENSIONS = ('.hdr', '.exr', '.pfm')


def is_valid_file(x):
    return has_file_allowed_extension(x, HDR_IMG_EXTENSIONS)


def is_same_config(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """Compare cfg1 with cfg2.

    Args:
        cfg1: Config
        cfg2: Config

    Returns:
        True if cfg1 == cfg2 else False

    """
    return cfg1 == cfg2


def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(cfg.pretty())
    print('--------------------')


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(classes: list[str], cfg: DictConfig) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.name == "itmnet":
        net = ITMNet(
            in_channels=1,
            num_classes=len(classes),
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            down_conv=nn.Conv2d,
            activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(1, 32, 32))
    else:
        raise NotImplementedError()

    return net
