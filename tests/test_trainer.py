import torch
from torchvision import datasets, transforms
import tempfile
import logging
from experiment_interface import Trainer, StopAtStep, SaveNetAtLast
from experiment_interface.nets import Conv2D

class MyCNN(torch.nn.Module):

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

def test_cifar10():

    net = MyCNN()

    logger = Trainer.get_logger()

    trnsfrms = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    cache_dir = tempfile.mkdtemp()
    logger.info('cache_dir: %s' % cache_dir) 
    train_dataset = datasets.CIFAR10(cache_dir, train=True, transform=trnsfrms, download=True)
    val_dataset = datasets.CIFAR10(cache_dir, train=False, transform=transforms.CenterCrop(28), download=True)

    result_dir = tempfile.mkdtemp()
    logger.info('result_dir: %s' % result_dir) 

    trainer = Trainer(
        net = net,
        train_dataset = train_dataset,
        batch_size = 64,
        loss_fn = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003 ),
        num_workers = 30,
        hooks = [StopAtStep(10), SaveNetAtLast(net_name='mycnn')],
        # num_workers = 60,
        # hooks = [StopAtStep(30000), SaveNetAtLast(net_name='mycnn')],
        result_dir = result_dir,
        log_file='train.log',

        )

    trainer.run()
