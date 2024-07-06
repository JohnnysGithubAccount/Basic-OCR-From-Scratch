import mlflow

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt.pyll import scope

import torch
from torch import nn

from tqdm import tqdm
from functools import partial
import os

from typing import Dict
from typing import Any

from dataset.dataset_utils import data_loader
from dataset.dataset_utils import get_classes
from dataset.dataset_utils import get_signature


from models.models import ResNet

from training.training_utils import train
from training.training_utils import validate

from utils.utils import accuracy_fn
from utils.utils import get_and_set_experiment
from utils.utils import ignore_warnings


def objective(
        params: Dict[str, Any],
        device: torch.device,
) -> float:
    in_planes = params["in_planes"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    optimizer = params["optimizer"]

    train_loader, test_loader = data_loader(batch_size=batch_size)

    model = ResNet(
        in_channels=1,
        layers=[3, 3, 3, 0],
        num_classes=len(get_classes()),
        in_planes=64,
        # in_planes=in_planes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with mlflow.start_run(nested=True, log_system_metrics=True) as run:
        mlflow.log_params(params=params)

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}")
            print("-" * 6)

            train_acc, train_loss = train(
                data_loader=train_loader,
                model=model,
                device=device,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                optimizer=optimizer
            )

            test_acc, test_loss = validate(
                data_loader=test_loader,
                model=model,
                device=device,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
            )

            mlflow.log_metric("Train acc", train_acc, step=epoch)
            mlflow.log_metric("Train loss", train_loss, step=epoch)
            mlflow.log_metric("Test acc", test_acc, step=epoch)
            mlflow.log_metric("Test loss", test_loss, step=epoch)

            print(f"Train acc: {train_acc}, Train loss: {train_loss}|Test_acc: {test_acc}, Test_loss: {test_loss}")

        model_signature = get_signature(
            dataloader=train_loader,
            device=device,
            model=model
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model.__class__.__name__,
            signature=model_signature
        )

        mlflow.set_tags(
            {
                "model_type": "classifier",
                "dataset": "MNIST + A_Z Handwritten Data",
                "purpose": "utilize the model"
            }
        )

        mlflow.set_tag(
            "mlflow.note.content",
            "This is a best version classifier for classifying the and written letters and digits dataset, yet?"
        )

        with open(f"{model.__class__.__name__}_summary.txt", "w") as file:
            file.write(str(model))

        mlflow.log_artifact(local_path=f"{model.__class__.__name__}_summary.txt",
                            artifact_path=f"{model.__class__.__name__}/artifacts")

    return -test_acc


def main():
    ignore_warnings()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

    run_name = "params_tunning"
    experiment_name = "handwritten_letters_digits_recognition"
    artifact_path = "hyper_tunning"

    search_space = {
        "in_planes": scope.int(hp.quniform('in_planes', 32, 128, 32)),
        'learning_rate': hp.loguniform('learning_rate', -5, -1),
        'batch_size': scope.int(hp.quniform('batch_size', 32, 128, 32)),
        'epochs': scope.int(hp.quniform('epochs', 5, 10, 5)),
        "optimizer": hp.choice("optimizer", ["adam", "sgd"])
    }

    get_and_set_experiment(
        experiment_name=experiment_name,
        artifact_path=artifact_path,
        tags={

        }
    )

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):

        trials = Trials()

        best_params = fmin(
            fn=partial(
                objective,
                device=device
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials
        )

        best_in_planes = int(best_params['in_planes'])
        best_learning_rate = best_params['learning_rate']
        best_batch_size = int(best_params['batch_size'])
        best_num_epochs = int(best_params['num_epochs'])
        best_optimizer = str(best_params["optimizer"])
        # best_accuracy = -trials.best_trial['result']['loss']

        model = ResNet(
            in_channels=1,
            layers=[3, 4, 6, 3],
            num_classes=len(get_classes()),
            in_planes=best_in_planes
        ).to(device)

        loss_fn = nn.CrossEntropyLoss()
        if best_optimizer == "adam":
            best_optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)
        else:
            best_optimizer = torch.optim.SGD(model.parameters(), lr=best_learning_rate)

        train_loader, test_loader = data_loader(batch_size=best_batch_size)

        for epoch in tqdm(range(best_num_epochs)):
            print(f"Testing epoch: {epoch}")
            train_acc, train_loss = train(
                data_loader=train_loader,
                model=model,
                device=device,
                loss_fn=loss_fn,
                optimizer=best_optimizer,
                accuracy_fn=accuracy_fn
            )

            print("-" * 6)

            test_acc, test_loss = validate(
                data_loader=test_loader,
                model=model,
                device=device,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn
            )

        model_signature = get_signature(
            dataloader=train_loader,
            device=device,
            model=model
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model.__class__.__name__,
            signature=model_signature
        )

        mlflow.log_metrics(metrics={
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss
        })

        mlflow.log_params(params=best_params)

        mlflow.set_tags(
            {
                "model_type": "classifier",
                "dataset": "MNIST + A_Z Handwritten Data",
                "status": "currently the best"
            }
        )

        mlflow.set_tag(
            "mlflow.note.content",
            "This is a best version classifier for classifying the and written letters and digits dataset"
        )

        with open(f"{model.__class__.__name__}_summary.txt", "w") as file:
            file.write(str(model))

        mlflow.log_artifact(local_path=f"artifacts/{model.__class__.__name__}_summary.txt",
                            artifact_path=f"{model.__class__.__name__}/artifacts")


if __name__ == "__main__":
    main()
