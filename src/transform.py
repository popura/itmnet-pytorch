from typing import Union, Optional

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

import hdrpy
from deepy.data.transform import Transform


class ReinhardTMO(Transform):
    """Wrapper for hdrpy.tmo.ReinhardTMO.
    Attributes:
        tmo: an instance of hdrpy.tmo.ReinhardTMO
    Examples:
    >>>
    """
    def __init__(
        self,
        ev: float = 0,
        mode: str = "global",
        whitepoint: Union[float, str] = "Inf") -> None:
        self.tmo = hdrpy.tmo.ReinhardTMO(
            ev=ev,
            mode=mode,
            whitepoint=whitepoint)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tmo(x.clone().detach().numpy().transpose((1, 2, 0)))
        return torch.from_numpy(x.astype(np.float32).transpose((2, 0, 1))).clone()


class KinoshitaITMO(Transform):
    """Wrapper for hdrpy.tmo.KinoshitaITMO.
    Attributes:
        itmo: an instance of hdrpy.tmo.KinoshitaITMO
    Examples:
    >>>
    """
    def __init__(
        self,
        alpha: Optional[float] = 0.18,
        hdr_gmean: Optional[float] = None) -> None:
        self.itmo = hdrpy.tmo.KinoshitaITMO(
            alpha=alpha,
            hdr_gmean=hdr_gmean)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.itmo(x.clone().detach().numpy().transpose((1, 2, 0)))
        return torch.from_numpy(x.astype(np.float32).transpose((2, 0, 1))).clone()


class RandomEilertsenTMO(Transform):
    """Wrapper for hdrpy.tmo.EilertsenTMO.
    Parameters `ev`, `exponent`, and `sigma`
    for hdrpy.tmo.EilertsenTMO are randomly determined,
    where `ev` follows an uniform distribution,
    and `exponent` and `sigma` follow normal distributions.
    Attributes:
        ev_range:
        exp_mean:
        exp_std:
        sigma_mean:
        sigma_std:
    Examples:
    >>>
    """
    def __init__(
        self,
        ev_range: tuple[int, int] = (-2, 2),
        exp_mean: float = 0.9,
        exp_std: float = 0.1,
        sigma_mean: float = 0.6,
        sigma_std: float = 0.1) -> None:
        super().__init__()
        self.ev_range = ev_range
        self.exp_mean = exp_mean
        self.exp_std = exp_std
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ev, exponent, sigma = self.get_params()
        tmo = hdrpy.tmo.EilertsenTMO(
            ev=ev,
            exponent=exponent,
            sigma=sigma)
        x = tmo(x.clone().detach().numpy().transpose((1, 2, 0)))
        return torch.from_numpy(x.astype(np.float32).transpose((2, 0, 1))).clone()
    
    def get_params(self) -> tuple[int, float, float]:
        ev = np.random.uniform(self.ev_range[0], self.ev_range[1])
        exponent = -1
        while exponent <= 0:
            exponent = np.random.normal(self.exp_mean, self.exp_std)

        sigma = -1
        while sigma <= 0:
            sigma = np.random.normal(self.sigma_mean, self.sigma_std)
        
        return ev, exponent, sigma


class ResizeToMultiple(Transform):
    def __init__(self, divisor, interpolation=Image.BILINEAR):
        super().__init__()
        self.divisor = int(divisor)
        self.interpolation = interpolation

    def __call__(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            new_size = (
                (img.height // self.divisor) * self.divisor,
                (img.width // self.divisor) * self.divisor)
        else:
            new_size = (torch.tensor(img.size()[-2:]) // self.divisor) * self.divisor
        return F.resize(img, list(new_size), self.interpolation)

    def __repr__(self, ):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(interpolation={0})'.format(interpolate_str)

