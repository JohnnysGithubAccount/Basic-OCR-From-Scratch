from models.models import ResNet
import torch
from torchvision import transforms
import cv2
from dataset.dataset_utils import get_classes
import matplotlib.pyplot as plt
from training.training_utils import validate
from dataset.dataset_utils import data_loader
from utils.utils import accuracy_fn


image_path = "images/test1.png"
image = cv2.imread(image_path)
image = cv2.resize(image, (28, 28))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
image = cv2.GaussianBlur(image, (5, 5), 0)
image = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])(image)

model = ResNet(
    in_channels=1,
    layers=[3, 4, 6, 3],
    num_classes=len(get_classes())
)
model.load_state_dict(torch.load("artifacts/model.pth"))
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
image = image.to(device)
plt.imshow(image.squeeze().cpu().detach().numpy(), cmap='gray')
plt.show()
labelNames = get_classes()


print(len(get_classes()))
print(image.shape)
print(image.unsqueeze(0).shape)

with torch.inference_mode():
    y_logit = model(image.unsqueeze(0))
    y_predict = torch.softmax(y_logit, dim=1)
    predicted_label = y_predict.argmax(dim=1)
    _, a = y_logit.max(1)
    print(y_logit)
    print(y_predict)
    print(predicted_label)
    print(labelNames[predicted_label])

