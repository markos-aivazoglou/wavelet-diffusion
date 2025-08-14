import os
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
from enum import Enum
from typing import Union
from torch.utils.data import Dataset


class DatasetName(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CELEBAHQ = "celebahq"
    STL10 = "stl10"

    @property
    def resolution(self) -> int:
        resolution = {
            self.MNIST: 28,
            self.CIFAR10: 32,
            self.STL10: 64,
            self.CELEBAHQ: 256,
        }
        return resolution[self]

    @property
    def channels(self) -> int:
        channels = {
            self.MNIST: 1,
            self.CIFAR10: 3,
            self.CELEBAHQ: 3,
            self.STL10: 3,
        }
        return channels[self]

    def __str__(self):
        return self.value


class CustomCIFAR10(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = datasets.CIFAR10(*args, **kwargs)
        self.resolution = 32
        self.channels = 3

    def __len__(self):
        return len(self.dataset)

    # return no labels
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img


class CustomSTL10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset = datasets.STL10(
            root=root,
            split="unlabeled" if train else "test",
            transform=transform,
            download=download,
        )
        self.resolution = 64
        self.channels = 3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img


class CelebAHQDataset(Dataset):
    def __init__(self, root, train=True, transform=None, **kwargs):
        """
        Args:
            img_dir (str): Directory with all the images
            split (str): Which split to use - 'train' or 'eval'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.img_dir = Path(os.path.join(root, "celeba_hq_256"))
        self.transform = transform
        self.resolution = 256
        self.channels = 3

        # Get all image files and sort them
        self.img_paths = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )

        # Split into train (first 25K) and eval (remaining 5K)
        if train:
            self.img_paths = self.img_paths[:25000]
        else:
            self.img_paths = self.img_paths[25000:]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


transforms = {
    "train": v2.Compose(
        [
            v2.RGB(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
            v2.RandomHorizontalFlip(),
        ]
    ),
    "eval": v2.Compose(
        [
            v2.RGB(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    ),
}


def load_datasets(
    dataset_name: DatasetName, root: str = "./data"
) -> tuple[Dataset, Dataset]:

    match dataset_name:
        case DatasetName.MNIST:
            return _load_datasets(datasets.MNIST, root)
        case DatasetName.CIFAR10:
            return _load_datasets(CustomCIFAR10, root)
        case DatasetName.CELEBAHQ:
            return _load_datasets(CelebAHQDataset, root)
        case DatasetName.STL10:
            transforms["train"].transforms.append(v2.Resize(64))
            transforms["eval"].transforms.append(v2.Resize(64))
            return _load_datasets(CustomSTL10, root)
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_datasets(dataset: Dataset, root: str) -> tuple[Dataset, Dataset]:
    train_dataset = dataset(
        root=root, train=True, download=True, transform=transforms["train"]
    )
    eval_dataset = dataset(
        root=root, train=False, download=True, transform=transforms["eval"]
    ) 
    return train_dataset, eval_dataset
