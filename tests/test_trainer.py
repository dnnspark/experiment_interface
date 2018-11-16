import torch
from torchvision import datasets, transforms
from experiment_interface import Trainer, StopAtStep, SaveNetAtLast
import tempfile
import logging

class MyCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_feat = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5, 2, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),

            torch.nn.Conv2d(256, 256, 3, 1, 1), # (256, 4, 4)
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256*4*4, 4, 1, 0),
            torch.nn.BatchNorm2d(256*4*4),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(256*4*4, 256, 1, 1, 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(256, 10, 1, 1, 0),
        )


    def forward(self, images):

        x = images
        x = self.conv_feat(x)
        x = self.mlp(x)
        x = torch.squeeze(x)

        return x


def test_cifar10():

    net = MyCNN()
    # import pdb; pdb.set_trace()

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
        log_file='train.log',

        )

    trainer.run()
