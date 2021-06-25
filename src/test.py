import argparse
from pathlib import Path
import typing

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
from torchinfo import summary

from omegaconf import DictConfig, OmegaConf

import hdrpy

from deepy.data.dataset import SelfSupervisedDataset
from deepy.data.vision.visiondataset import UnorganizedImageFolder
import deepy.data.transform
import deepy.data.vision.transform
from deepy.train.trainer import RegressorTrainer
from deepy.train.extension import (
    IntervalTrigger,
    MinValueTrigger,
    MaxValueTrigger,
    ModelSaver,
    HistorySaver
)

import transform as mytransform
import util as myutil
import train_id as tid


def get_transform(cfg: DictConfig):
    pre_transforms = None

    dict_cfg_crop = cfg.dataset.transforms.paired_random_resized_crop.param
    if dict_cfg_crop.interpolation == 'Image.BICUBIC':
        interpolation = Image.BICUBIC

    transforms = deepy.data.transform.PairedCompose([
        deepy.data.transform.ToPairedTransform(
            torchvision.transforms.ToTensor()),
        deepy.data.vision.transform.PairedPowerTwoResize(interpolation=interpolation),
        deepy.data.transform.SeparatedTransform(
            transform=mytransform.ReinhardTMO(),
            target_transform=mytransform.RandomEilertsenTMO()),
    ])

    return pre_transforms, transforms


def get_dataset(cfg: DictConfig):
    cwd = Path.cwd()
    p = Path(cwd) / cfg.dataset.path

    testset = SelfSupervisedDataset(
        UnorganizedImageFolder(
            str(p / "test"),
            pre_load=False,
            loader=hdrpy.io.read, is_valid_file=myutil.is_valid_file)
    )

    return testset


def predict(path, dataset, net, device):
    p = Path(path)
    (p / 'imgs').mkdir(parents=True, exist_ok=True)
    transforms = deepy.data.transform.Compose(
        [torchvision.transforms.ToTensor(),
         mytransform.ResizeToMultiple(divisor=16, interpolation=Image.BICUBIC),
         mytransform.RandomEilertsenTMO()])
    
    post_transforms = mytransform.KinoshitaITMO()
    target_transforms = deepy.data.transform.Compose(
        [torchvision.transforms.ToTensor(),
         mytransform.ReinhardTMO()])

    with torch.no_grad():
        for i in range(len(dataset)):
            q = dataset.dataset.samples[i]
            q = Path(q)

            print('{:04d}/{:04d}: File Name {}'.format(i, len(dataset), q.name))

            sample, hdr_target = dataset[i]
            sample = transforms(sample).unsqueeze(0).float().to(device)
            ldr_predict = net(sample).to('cpu').clone().detach().squeeze(0)
            hdr_predict = post_transforms(torch.clamp(ldr_predict, 0, 1))
            height, width, _ = hdr_target.shape
            sample = sample.to('cpu').clone().detach().squeeze(0)
            sample = F.resize(sample, (height, width), Image.BICUBIC)
            ldr_predict = F.resize(ldr_predict, (height, width), Image.BICUBIC)
            hdr_predict = F.resize(hdr_predict, (height, width), Image.BICUBIC)
            ldr_target = target_transforms(hdr_target)

            hdrpy.io.write(
                p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_input.jpg'),
                sample.clone().detach().numpy().transpose((1, 2, 0)))
            hdrpy.io.write(
                p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_output.jpg'),
                ldr_predict.clone().detach().numpy().transpose((1, 2, 0)))
            hdrpy.io.write(
                p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_prediction.pfm'),
                hdr_predict.clone().detach().numpy().transpose((1, 2, 0)))
            hdrpy.io.write(
                p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_reinhard.jpg'),
                ldr_target.clone().detach().numpy().transpose((1, 2, 0)))
            hdrpy.io.write(
                p / 'imgs' / ('{:04d}'.format(i) + '_' + q.stem + '_target.pfm'),
                hdr_target)


def main(cfg: DictConfig, train_id:str) -> None:
    cwd = Path.cwd()
    myutil.print_config(cfg)
    # Setting seed 
    myutil.set_random_seed(0)

    model_file_name = "{}_best.pth".format(cfg.model.name)
    
    # Checking history directory
    history_dir = cwd / 'history' / train_id
    if (history_dir / model_file_name).exists():
        pass
    else:
        return

    # Setting result directory
    # All outputs will be written into (p / 'result' / train_id).
    if not (cwd / 'result').exists():
        (cwd / 'result').mkdir(parents=True)
    result_dir = cwd / 'result' / train_id
    if result_dir.exists():
        # removing files in result_dir?
        pass
    else:
        result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing
    testset = get_dataset(cfg)
    net = myutil.get_model(cfg)
    net.module.load_state_dict(torch.load(str(history_dir / model_file_name)))
    predict(result_dir, testset, net, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default='./history',
                        help='Directory path for searching trained models')
    args = parser.parse_args()
    p = Path.cwd() / args.history_dir
    for q in p.glob('**/config.yaml'):
        cfg = OmegaConf.load(str(q))
        cfg.loader.num_workers = 0
        train_id = q.parent.name
        main(cfg, train_id)
