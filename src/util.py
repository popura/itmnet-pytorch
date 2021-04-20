import typing

import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary

from omegaconf import DictConfig

from deepy.data.dataset import has_file_allowed_extension
from deepy.nn.model import UNet

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


def get_model(cfg: DictConfig) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.name == "itmnet":
        net = ITMNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=nn.ConvTranspose2d,
            down_conv=nn.Conv2d,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(3, 256, 256))
    elif cfg.model.name == "unet":
        net = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            up_conv=nn.ConvTranspose2d,
            down_conv=nn.Conv2d,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
            final_activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(3, 256, 256))
    else:
        raise NotImplementedError()

    return net
