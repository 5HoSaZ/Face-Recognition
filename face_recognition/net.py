# import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv_1_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_1_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv_4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )

        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=13 * 13 * 256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Convo-block 1
        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.maxpool(x)
        # Convo-block 2
        x = self.relu(self.conv_2(x))
        x = self.maxpool(x)
        # Convo-block 3
        x = self.relu(self.conv_3(x))
        x = self.maxpool(x)
        # Convo-block 4
        x = self.relu(self.conv_4(x))
        x = self.maxpool(x)
        # Flatten
        x = self.flatten(x)
        # Dense
        x = self.relu(self.fc1(x))
        x = nn.Dropout(0.2)(x)
        x = self.relu(self.fc2(x))
        x = nn.Dropout(0.2)(x)
        x = self.softmax(x)
        return x
