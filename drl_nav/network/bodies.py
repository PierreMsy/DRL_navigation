from torch import nn
import torch.nn.functional as F

class ConvBody(nn.Module):
    """
    Convolutional network that convert pixel state into features.
    """
    
    def __init__(self, in_channels=3):
        super(ConvBody, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))

        output = x.view(x.size(0), -1)

        return output