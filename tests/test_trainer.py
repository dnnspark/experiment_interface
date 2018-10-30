import torch
from torchvision import datasets, transforms
from experiment_interface import Trainer, StopAtStep

class MyCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
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

        self.fc_layers = torch.nn.Sequential(
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
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = torch.squeeze(x)

        return x


def test_cifar10():

    net = MyCNN()

    trnsfrms = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10('./data/', train=True, transform=trnsfrms, download=True)

    val_dataset = datasets.CIFAR10('./data/', train=False, transform=transforms.CenterCrop(28), download=True)

    trainer = Trainer(
        model = net,
        train_dataset = train_dataset,
        batch_size = 64,
        loss_fn = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003 ),
        num_workers = 30,
        hooks = [StopAtStep(10)],
        log_file='train.log',

        )

    trainer.run()


# if __name__ == '__main__':
#     test_cifar10()

