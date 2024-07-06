import torch

from torchvision import transforms

from models.models import ResNet

from dataset.dataset_utils import get_classes

import cv2

import imutils
from imutils.contours import sort_contours

import matplotlib.pyplot as plt


def implementing(image_path: str, model: torch.nn.Module, device):
    # Original image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    plt.imshow(plt.imread(image_path), cmap="viridis")
    plt.title("Original")
    plt.show()
    print(image.shape)
    # Gray - reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray scale")
    plt.show()
    # Blurred
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.imshow(blurred, cmap="gray")
    plt.title("Blurred")
    plt.show()
    # Edge detection map
    edged = cv2.Canny(blurred, 5, 100)
    plt.imshow(edged, cmap="gray")
    plt.title("Edged")
    plt.show()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0]
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method='left-to-right')[0]

    temp_img = image.copy()
    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours", temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    chars = list()

    for c in cnts:
        # Compute the box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        if (5 <= w <= 250) and (15 <= h <= 250):
            roi = blurred[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            else:
                thresh = imutils.resize(thresh, height=28)

            (tH, tW) = thresh.shape
            dX = int(max(14, 28 - tW) / 2.0)
            dY = int(max(14, 28 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))

            padded = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])(padded).to(device)
            chars.append((padded, (x, y, w, h)))

    filtered_chars = []
    for i, (img, box_a) in enumerate(chars):
        x1_a, y1_a, w_a, h_a = box_a
        is_inside_other_box = False

        for j, (_, box_b) in enumerate(chars):
            if i != j:  # Skip comparing a box with itself
                x1_b, y1_b, w_b, h_b = box_b
                if (x1_a >= x1_b) and (y1_a >= y1_b) and (x1_a + w_a <= x1_b + w_b) and (y1_a + h_a <= y1_b + h_b):
                    is_inside_other_box = True
                    break  # Box A is inside Box B
        if not is_inside_other_box:
            filtered_chars.append((img, box_a))

    boxes = [b[1] for b in filtered_chars]
    chars = [c[0] for c in filtered_chars]

    predictions = [
        torch.softmax(model(char.unsqueeze(0)), dim=1) for char in chars
    ]

    labelNames = get_classes()

    for letter_index, (prediction, (x, y, w, h)) in enumerate(zip(predictions, boxes)):
        i = prediction.argmax(dim=1)
        prob = prediction.max(dim=1).values
        label = labelNames[i.item()]

        print(f"[{letter_index}][INFOR] {label} - {prob.item() * 100:.2f}%")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Image", image)
    cv2.waitKey(0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet(
        in_channels=1,
        layers=[3, 4, 6, 3],
        num_classes=36
    ).to(device)
    state_dict = torch.load(r"artifacts/model.pth")
    model.load_state_dict(state_dict)
    model.eval()

    image_dir = r"images/test6.png"

    implementing(
        image_path=image_dir,
        model=model,
        device=device
    )


if __name__ == "__main__":
    main()
