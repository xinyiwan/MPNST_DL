import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *


class cnn3d(nn.Module):

    def __init__(self):
        super(cnn3d, self).__init__()

        # First 3D convolutional layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Second 3D convolutional layer
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(16 * 32 * 32 * 16, 256)  # Adjust the input size based on your data
        self.relu3 = nn.ReLU()

        # Output layer
        self.output_layer = nn.Linear(256, 2)  # Set num_classes to the number of output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output before passing it to the fully connected layer
        x = x.view(-1, 16 * 32 * 32 * 16)  # Adjust the size based on your data and architecture
        x = self.relu3(self.fc(x))

        # Output layer
        x = self.output_layer(x)

        return x

