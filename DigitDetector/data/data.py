# ./datasets/data.py
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
import os
import numpy
import pandas
import yaml
import random

def get_data_loaders(config_path="../config.yaml"):
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    batch_size = config['dataset']['batch_size']

    train_dataset = datasets.MNIST(root='dataset/', download=True, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='dataset/', download=True, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader