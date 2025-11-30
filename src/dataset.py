import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_datasets():
    # Train Data with ToTensor
    train_data = datasets.FashionMNIST(
        root="/data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # Compute mean & std
    all_pixels = torch.stack([pix for pix, _ in train_data])
    data_mean = torch.mean(all_pixels)
    data_std = torch.std(all_pixels)

    # Transformations
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((data_mean,), (data_std,))
    ])

    # Transformed train & test data
    transformed_train_data = datasets.FashionMNIST(
        root='/transformed_data',
        train=True,
        transform=transformation,
        download=True
    )
    transformed_test_data = datasets.FashionMNIST(
        root='/transformed_data',
        train=False,
        transform=transformation,
        download=True
    )

    # Split train into train/val
    train_len = int(0.9 * len(transformed_train_data))
    val_len = len(transformed_train_data) - train_len
    train_subset, val_subset = random_split(transformed_train_data, [train_len, val_len])

    # Dataloaders
    train_batch = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_batch = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_batch = DataLoader(transformed_test_data, batch_size=64, shuffle=False)

    return train_batch, val_batch, test_batch
