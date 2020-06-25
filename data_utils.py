# Utility functions for the real NVP model.
#
# Author: Fangzhou Mu <fmu2@wisc.edu>
#
# https://github.com/fmu2/realNVP

"""Utility functions for real NVP.
"""

import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# DATA_ROOT = "../../data/"
DATA_ROOT = "~/datasets/"


class Hyperparameters():
    def __init__(self, base_dim, res_blocks, bottleneck, 
        skip, weight_norm, coupling_bn, affine):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine

class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def load(dataset):
    """Load dataset.

    Args:
        dataset: name of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    if dataset == 'cifar10':    # 3 x 32 x 32
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.CIFAR10(DATA_ROOT + 'CIFAR10', 
            train=True, download=True, transform=transform)
        [train_split, val_split] = data.random_split(train_set, [46000, 4000])

    elif dataset == 'celeba':   # 3 x 218 x 178
        data_info = DataInfo(dataset, 3, 64)
        def CelebACrop(images):
            return transforms.functional.crop(images, 40, 15, 148, 148)
        transform = transforms.Compose(
            [CelebACrop, 
             transforms.Resize(64), 
             transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.ImageFolder(DATA_ROOT + 'CelebA/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [150000, 12770])

    elif dataset == 'imnet32':
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder(DATA_ROOT + 'ImageNet32/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    elif dataset == 'imnet64':
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder(DATA_ROOT + 'ImageNet64/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    return train_split, val_split, data_info

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))
