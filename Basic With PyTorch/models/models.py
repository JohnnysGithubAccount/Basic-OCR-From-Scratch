import torch.cuda
from torch import nn
from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 down_sample: nn.Module | None = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.down_sample = down_sample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels: int,
                 layers: List[int],
                 num_classes: int,
                 block=ResidualBlock,
                 in_planes: int = 64):
        super(ResNet, self).__init__()

        self.in_planes = in_planes
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.in_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer0 = self._make_layer(block, self.in_planes, layers[0], stride=1)
        self.layer1 = self._make_layer(block, self.in_planes * 2, layers[1], stride=2)
        self.layer2 = self._make_layer(block, self.in_planes * 4, layers[2], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def _make_layer(self, block, planes, blocks, stride: int = 1):
        down_sample = None
        if stride != 1 or self.in_planes != planes:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes)
            )
        layers = list()
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet(
        in_channels=1,
        layers=[3, 4, 6, 3],
        num_classes=10
    ).to(device)

    print(model)


if __name__ == "__main__":
    main()
