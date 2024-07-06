import torch

from models.models import ResNet

from dataset.dataset_utils import get_classes
from dataset.dataset_utils import data_loader
from dataset.dataset_utils import get_labels

from training.training_utils import validate
from training.training_utils import get_test_predictions

from utils.utils import accuracy_fn

import cv2

from imutils import build_montages

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def display_confusion_matrix(dataloader, model, device, label_names):
    y_true = get_labels(dataloader)
    y_predictions = get_test_predictions(
        data_loader=dataloader,
        model=model,
        device=device
    )
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_predictions.cpu().detach().numpy(),
        # labels=label_names
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax)
    plt.savefig("artifacts/confusion_matrix.png")
    plt.show()


def val(dataloader, model, device):
    print("[INFO] validating model")
    test_acc, test_loss = validate(
        data_loader=dataloader,
        model=model,
        device=device,
        loss_fn=torch.nn.CrossEntropyLoss(),
        accuracy_fn=accuracy_fn
    )
    print(f"Model's acc: {test_acc}, loss: {test_loss}")


def plot_vals(dataloader, model, device, label_names):
    images = list()
    side = 10

    for i, (img, label) in enumerate(dataloader):
        if i >= side * side:
            break
        img, label = img.to(device), label.to(device)

        with torch.inference_mode():
            probs = model(img)
            prediction = probs.argmax(dim=1).item()

            label_name = label_names[prediction]

            image = (img.squeeze().cpu().detach().numpy() * 255).astype("uint8")
            color = (0, 255, 0) if prediction == label.item() else (0, 0, 255)

            image = cv2.merge([image] * 3)
            image = cv2.resize(image, (75, 75), interpolation=cv2.INTER_LINEAR)
            cv2.putText(image, label_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            images.append(image)

    montage = build_montages(images, (75, 75), (side, side))[0]

    cv2.imshow("OCR Results", montage)
    cv2.waitKey(0)


def main():
    labelNames = get_classes()
    print(labelNames)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    model = ResNet(
        in_channels=1,
        layers=[3, 4, 6, 3],
        num_classes=len(get_classes())
    ).to(device)

    print("[INFO] loading weights")
    state_dict = torch.load(r"artifacts/model.pth")
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    train_loader, test_loader = data_loader(batch_size=32)
    val(
        dataloader=test_loader,
        model=model,
        device=device,
        # label_names=labelNames
    )

    # _, test_loader = data_loader(batch_size=128)
    # display_confusion_matrix(
    #     dataloader=test_loader,
    #     model=model,
    #     device=device,
    #     label_names=labelNames
    # )


if __name__ == "__main__":
    main()
