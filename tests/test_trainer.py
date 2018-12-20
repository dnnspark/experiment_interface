import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import tempfile
import logging
from experiment_interface import Trainer
from experiment_interface.common import DebugMode
from experiment_interface.logger import get_train_logger
from experiment_interface.hooks import StopAtStep
from experiment_interface.nets import Conv2D
from experiment_interface.evaluator.metrics import ClassificationAccuracy
from experiment_interface.hooks import ValidationHook
from experiment_interface.hooks import ValLossAccViz
from experiment_interface.hooks.viz_runner import VisdomRunner
# from experiment_interface.hooks.viz_runner import VisdomRunner
# from experiment_interface.hooks.viz_runner import plot_val_lossacc

class MyCNN(torch.nn.Module):

    _name = 'mycnn'

    def __init__(self):
        super().__init__()

        self.feature_extractor = torch.nn.Sequential(
            Conv2D(3,    64, 5, 2, use_batchnorm=True, act_fn='relu'),

            Conv2D(64,   64, 3, 1, use_batchnorm=True, act_fn='relu'),
            Conv2D(64,   64, 3, 1, use_batchnorm=True, act_fn='relu'),
            Conv2D(64,  128, 3, 2, use_batchnorm=True, act_fn='relu'),

            Conv2D(128, 128, 3, 1, use_batchnorm=True, act_fn='relu'),
            Conv2D(128, 128, 3, 1, use_batchnorm=True, act_fn='relu'),
            Conv2D(128, 256, 3, 2, use_batchnorm=True, act_fn='relu'),

            Conv2D(256, 256, 3, 1, use_batchnorm=True, act_fn='relu'),
            Conv2D(256, 256, 3, 1, use_batchnorm=True, act_fn='relu'), # (256, 4, 4)
            )

        self.mlp = torch.nn.Sequential(
            Conv2D(256, 256*4, 4, 1, padding=0),
            Conv2D(256*4, 128, 1, 1, padding=0),
            Conv2D(128, 10, 1, 1, padding=0),
        )


    def forward(self, images):
        x = images
        x = self.feature_extractor(x)
        x = self.mlp(x)
        x = torch.squeeze(x)

        return x

CATEGORY_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

class Cifar10TrainDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None, download=False):
        cifar10_train_dataset = datasets.CIFAR10(root, True, transform, target_transform, download)
        self.dataset = torch.utils.data.Subset(cifar10_train_dataset, np.arange(0,49600))

    def __getitem__(self, index):
        image, label = self.dataset[index]
        img_id = 'train_%05d' % index 
        return {
            'inputs': image,
            'labels': label,
            'metadata':
                {'img_ids': img_id },
            }

    def __len__(self):
        return len(self.dataset)

class Cifar10ValDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None, download=False):
        cifar10_train_dataset = datasets.CIFAR10(root, True, transform, target_transform, download)
        self.dataset = torch.utils.data.Subset(cifar10_train_dataset, np.arange(49600,50000))

    def __getitem__(self, index):
        image, label = self.dataset[index]
        img_id = 'train_%05d' % index 
        return {
            'inputs': image,
            'labels': label,
            'metadata':
                {'img_ids': img_id },
            }

    def __len__(self):
        return len(self.dataset)

class Cifar10TestDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.dataset = datasets.CIFAR10(root, False, transform, target_transform, download)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        img_id = 'test_%05d' % index 
        return {
            'inputs': image,
            'labels': label,
            'metadata':
                {'img_ids': img_id },
            }

    def __len__(self):
        return len(self.dataset)


def most_probable_class(logits):
    '''
    Input
    =====
        logits: (B, C) Tensor
    '''
    return torch.argmax(logits, dim=1)

import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def reduce_ticklabelsize(ax, axes=['x'], factor=.5):
    if 'x' in axes:
        ticklabels = ax.get_xticklabels()
        for label in ticklabels:
            label.set_size(label.get_size() * factor)
        ax.set_xticklabels(ticklabels)
    if 'y' in axes:
        ticklabels = ax.get_yticklabels()
        for label in ticklabels:
            label.set_size(label.get_size() * factor)
        ax.set_yticklabels(ticklabels)


class ConfMatViz(VisdomRunner):


    def refresh(self, context, ax):
        val_dir = os.path.join( context.trainer.result_dir, 'val', 'val_acc')
        all_val_files = glob.glob(os.path.join( val_dir, '*.csv'))
        if len(all_val_files) == 0:
            return None

        latest = max(all_val_files, key=os.path.getctime)
        df = pd.read_csv(latest, index_col=0)

        classacc_metric = context.trainer.valhook_tab['val_acc'].metric
        conf_mat = classacc_metric.confusion_mat(df)

        # ax = plt.gca()
        sns.heatmap(data=conf_mat, cmap='Greys_r', ax=ax)
        reduce_ticklabelsize(ax, ['x','y'], factor=.7)
        ax.set_xlabel('groundtruth')
        ax.set_ylabel('prediction')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

def test_cifar10():

    net = MyCNN()

    logger = get_train_logger()

    train_trnsfrms = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    val_trnsfrms = transforms.Compose([
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        ])

    # cache_dir = tempfile.mkdtemp()
    # cache_dir = '/var/folders/_1/9y4khvtd4sbbpf0wz_8fzlq00000gn/T/tmpktj6vddq'
    cache_dir = '/cluster/storage/dpark/cifar10/'
    logger.info('cache_dir: %s' % cache_dir) 
    train_dataset = Cifar10TrainDataset(cache_dir, transform=train_trnsfrms, download=True)
    val_dataset = Cifar10ValDataset(cache_dir, transform=val_trnsfrms, download=False)

    result_dir = tempfile.mkdtemp()
    logger.info('result_dir: %s' % result_dir) 

    trainer = Trainer(
        net = net,
        train_dataset = train_dataset,
        batch_size = 64,
        loss_module_class = torch.nn.CrossEntropyLoss,
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003 ),
        result_dir = result_dir,
        log_interval = 10,
        num_workers = 30,
        max_step = 30000,
        val_dataset = val_dataset,
        val_interval = 200,
        )

    class_acc_metric = ClassificationAccuracy(category_names = CATEGORY_NAMES)
    accuracy_valhook = ValidationHook(
        dataset = val_dataset, 
        interval = 200, 
        name = 'val_acc', 
        predict_fn = most_probable_class, 
        metric = class_acc_metric,
        save_best=False,
        )

    trainer.register_val_hook(accuracy_valhook)

    val_lossacc_viz = ValLossAccViz(env='val')
    trainer.register_viz_hook(val_lossacc_viz)

    confmat_viz = ConfMatViz(env='val')
    trainer.register_viz_hook(confmat_viz)



    trainer.run(debug_mode=DebugMode.DEBUG)
    # trainer.run(debug_mode=DebugMode.DEV)
