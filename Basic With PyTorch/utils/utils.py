import torch

import warnings

import mlflow

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict
from typing import Any
from typing import List

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def plot_filters(image: torch.Tensor, rows: int = 1, cols: int = 1) -> None:
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(rows * 2 + 1, cols * 2 + 1))
    if rows != 1 and cols != 1:
        axes = axes.flatten()
        for i in range(rows * cols):
            filters = image.squeeze(0)[i].cpu().detach().numpy()
            axes[i].imshow(filters, cmap='gray')
            axes[i].axis("off")
    else:
        filters = image.squeeze(0)[0].cpu().detach().numpy()
        axes.imshow(filters, cmap='gray')
        axes.axis("off")
    plt.tight_layout()
    plt.show()
    

def accuracy_fn(y_predict: torch.Tensor, y_true: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_predict).sum().item()
    acc = correct * 100 / len(y_true)
    return acc


def get_and_set_experiment(experiment_name: str,
                           artifact_path: str,
                           tags: Dict[str, str]) -> None:
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_path,
            tags=tags
        )
    except:
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name
        ).experiment_id
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)


def ignore_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def main():
    pass


if __name__ == "__main__":
    main()
