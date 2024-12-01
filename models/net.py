# import torch
import torch.nn as nn


class PastNet(nn.Module):
    def __init__(self, num_classes):
        """Past model that from previous project"""
        super(PastNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=13 * 13 * 256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.dropout = nn.Dropout(0.65)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Convo-block 1
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.batchnorm1(self.relu(self.conv2(x)))
        x = self.maxpool(x)
        # Convo-block 2
        x = self.batchnorm2(self.relu(self.conv3(x)))
        x = self.batchnorm2(self.relu(self.conv4(x)))
        x = self.maxpool(x)
        # Convo-block 3
        x = self.batchnorm3(self.relu(self.conv5(x)))
        x = self.batchnorm3(self.relu(self.conv6(x)))
        x = self.maxpool(x)
        # Convo-block 4
        x = self.batchnorm4(self.relu(self.conv7(x)))
        x = self.batchnorm4(self.relu(self.conv8(x)))
        x = self.maxpool(x)
        # Flatten
        x = self.flatten(x)
        # Dense
        x = self.batchnorm5(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x


class ConvolutionNet(nn.Module):
    def __init__(
        self, num_classes: int, hidden_features: int = 1024, drop: float = 0.2
    ):
        super(ConvolutionNet, self).__init__()

        self.block1 = DoubleConvBlock(3, 16)
        self.block2 = DoubleConvBlock(16, 32)
        self.block3 = DoubleConvBlock(32, 64)
        self.block4 = DoubleConvBlock(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * 13 * 128, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.lgsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # Convolution blocks
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.block4(x)
        x = self.maxpool(x)
        # Flatten
        x = self.flatten(x)
        # Dense
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Softmax
        x = self.lgsoftmax(x)
        return x
