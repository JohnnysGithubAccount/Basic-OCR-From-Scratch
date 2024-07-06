import torch

from tqdm import tqdm

from dataset.dataset_utils import data_loader
from dataset.dataset_utils import get_classes

from models.models import ResNet

from training.training_utils import train
from training.training_utils import validate

from utils.utils import accuracy_fn


def main():

    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = data_loader(batch_size=32)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ResNet(
        in_channels=1,
        layers=[3, 4, 6, 3],
        num_classes=len(get_classes())
    )

    state_dict = torch.load(f"artifacts/model1.pth")
    model.load_state_dict(state_dict)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()

    # epochs = 5
    epochs = 2

    for epoch in tqdm(range(epochs)):

        print(f"Epoch: {epoch}/{epochs}")

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
            accuracy_fn=accuracy_fn
        )

        print(f"Train_acc: {train_acc}, Train_loss: {train_loss}")
        print(f"Test_acc: {test_acc}, Test_loss: {test_loss}")

    torch.save(model.state_dict(), f='artifacts/model1.pth')


if __name__ == "__main__":
    main()
