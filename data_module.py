import torch

import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

class Reshape(torch.nn.Module):
    def __call__(self, tensor):
        tensor = tensor.view(-1, 3, 32, 32)
        return tensor.squeeze()
dataset_path = r'C:\Users\Jasper\Documents\Coding'
class SigmoidTransform:
    """Custom transformation to apply the sigmoid function to a tensor."""
    def __call__(self, tensor):
        return torch.sigmoid(tensor)
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=dataset_path, batch_size=64):

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (3, 32, 32)
        #self.transform = transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=1)
        CIFAR10(self.data_dir, train=False, download=1)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            CIFAR10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.CIFAR10_train, self.CIFAR10_val = random_split(CIFAR10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.CIFAR10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.CIFAR10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.CIFAR10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.CIFAR10_test, batch_size=self.batch_size)

module = CIFAR10DataModule(data_dir=dataset_path)
module.setup()