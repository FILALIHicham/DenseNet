import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

# Dataset mean and std values
MEAN_STD = {
    'CIFAR10': {
        'mean': [125.3 / 255, 123.0 / 255, 113.9 / 255],
        'std': [63.0 / 255, 62.1 / 255, 66.7 / 255]
    },
    'CIFAR100': {
        'mean': [129.3 / 255, 124.1 / 255, 112.4 / 255],
        'std': [68.2 / 255, 65.4 / 255, 70.4 / 255]
    },
    'SVHN': {
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970]
    }
}

def get_transforms(dataset_name, augment):
    """Returns appropriate transforms based on dataset and augmentation flag."""

    mean_std = MEAN_STD[dataset_name]

    normalize = transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])

    if augment and dataset_name in ['CIFAR10', 'CIFAR100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # No augmentation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # Test transform (always no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def get_dataloaders(dataset_name, batch_size=64, augment=True, data_dir='./data', num_workers=2):
    dataset_name = dataset_name.upper()
    train_transform, test_transform = get_transforms(dataset_name, augment)

    if dataset_name == 'CIFAR10':
        train_set = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    elif dataset_name == 'CIFAR100':
        train_set = datasets.CIFAR100(root=data_dir, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)

    elif dataset_name == 'SVHN':
        # SVHN dataset splits: 'train', 'test', and 'extra'
        train_set = datasets.SVHN(root=data_dir, split='train', transform=train_transform, download=True)
        extra_set = datasets.SVHN(root=data_dir, split='extra', transform=train_transform, download=True)

        # Concatenate the train and extra datasets into one dataset
        train_set = ConcatDataset([train_set, extra_set])

        test_set = datasets.SVHN(root=data_dir, split='test', transform=test_transform, download=True)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader