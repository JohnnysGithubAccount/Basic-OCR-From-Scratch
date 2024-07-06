import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from typing import Tuple
from typing import Any

from tqdm import tqdm


def train(data_loader: DataLoader,
          model: nn.Module,
          device: torch.device,
          loss_fn: Any,
          accuracy_fn: Any,
          optimizer: torch.optim) -> Tuple[float, torch.tensor]:
    model.to(device)
    model.train()

    train_loss = train_acc = 0

    scaler = GradScaler()

    for batch, (X, y) in enumerate(data_loader):

        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        X, y = X.to(device), y.to(device)

        with autocast():
            y_logit = model(X)
            y_predict = torch.softmax(y_logit, dim=1).argmax(dim=1)

            loss = loss_fn(y_logit, y)
            acc = accuracy_fn(y_predict, y)

        train_loss += loss
        train_acc += acc

        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()

        if batch % 500 == 0:
            print(f"Training batch: {batch}/{len(data_loader)}")

    train_acc /= len(data_loader)
    train_loss /= len(data_loader)

    return train_acc, train_loss


def validate(data_loader: DataLoader,
             model: nn.Module,
             device: torch.device,
             loss_fn: Any,
             accuracy_fn: Any
             ) -> Tuple[float, torch.tensor]:
    model.to(device)
    model.eval()

    test_loss = test_acc = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            y_logit = model(X)
            y_predict = torch.softmax(y_logit, dim=1).argmax(dim=1)

            test_loss += loss_fn(y_logit, y)
            test_acc += accuracy_fn(y_predict, y)

            if batch % 500 == 0:
                print(f"Testing batch: {batch}/{len(data_loader)}")

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_acc, test_loss


def get_test_predictions(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.device
) -> torch.Tensor:
    y_predictions = list()

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc='Making decisions'):
            X, y = X.to(device), y.to(device)

            y_logit = model(X)
            y_prediction = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_predictions.append(y_prediction.cpu())

    return torch.cat(y_predictions)


def main():
    pass


if __name__ == "__main__":
    main()
