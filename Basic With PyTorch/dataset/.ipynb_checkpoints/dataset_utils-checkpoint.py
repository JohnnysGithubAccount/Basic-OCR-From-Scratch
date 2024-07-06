import matplotlib.pyplot as plt

import pandas as pd

from typing import Tuple
from typing import List

from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models.signature import infer_signature

import torch

import torchvision
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import ToPILImage
from torchvision.transforms import Resize
from torchvision.transforms import Normalize


class HandwrittenAZDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: torchvision.transforms = None):
        self.dataframe = df
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx, 1:].values.astype('uint8').reshape((28, 28, 1))
        label = self.dataframe.iloc[idx, 0] + 10
        if self.transform:
            image = self.transform(image)

        return image, label


def loading_data(
        root: str = r"D:\UsingSpace\Programing Languages Learning\Python Programing\HCMUTE Projects\Practical Python "
                    r"Programming\Final Project\dataset\data",
        csv_file: str = r"D:\UsingSpace\Programing Languages Learning\Python Programing\HCMUTE Projects\Practical "
                        r"Python Programming\Final Project\dataset\data\A_Z Handwritten Data.csv"
) -> Tuple[Dataset, Dataset]:

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
        target_transform=None
    )

    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
        target_transform=None
    )

    az_data = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(az_data, test_size=0.2, random_state=42)

    transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    train_az = HandwrittenAZDataset(
        df=train_df,
        transform=transform
    )

    test_az = HandwrittenAZDataset(
        df=test_df,
        transform=transform
    )

    combined_train_dataset = ConcatDataset([train_data, train_az])
    combined_test_dataset = ConcatDataset([test_data, test_az])

    return combined_train_dataset, combined_test_dataset


def data_loader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    train_data, test_data = loading_data()

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, test_loader


def get_classes() -> List[str]:
    mnist_classes = [str(i) for i in range(10)]

    az_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    combined_classes = mnist_classes + az_classes

    return combined_classes


def plot_images(dataloader: DataLoader, class_names: List[str]) -> None:
    torch.manual_seed(42)

    side = 5
    num_images = side * side

    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(side, side, figsize=(9, 9))
    axes = axes.flatten()

    for i in range(num_images):
        rand_idx = torch.randint(0, len(images), size=[1]).item()

        img, label = images[rand_idx].numpy().squeeze(), labels[rand_idx].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(class_names[label])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def get_labels(dataloader: DataLoader) -> List[int]:
    labels = list()

    for _, label in dataloader:
        labels.extend(label.tolist())

    return labels


def get_signature(dataloader: DataLoader, device: torch.device, model: torch.nn.Module) -> mlflow.models.ModelSignature:
    examples = next(iter(dataloader))

    inputs = examples[0].to(device)
    outputs = model(inputs)

    model_signature = infer_signature(
        inputs.cpu().detach().numpy(),
        outputs.cpu().detach().numpy()
    )

    return model_signature


def main():
    train_loader, test_loader = data_loader(batch_size=32)
    plot_images(train_loader, get_classes())


if __name__ == "__main__":
    main()
