from torch import nn
import torch.nn.functional as F

from drl_nav.network.utils import compute_output_size


class ConvBody(nn.Module):
    """
    Convolutional network that convert pixel state into features.
    """
    def __init__(self, in_channels=3, input_size=84, **kwargs):
        super(ConvBody, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_size = compute_output_size(self, input_size, in_channels)

    def forward(self, x):

        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))

        output = x.view(x.size(0), -1)

        return output
    
class DummyBody(nn.Module):
    """
    Empty body used to test the head.
    """
    def __init__(self, input_size, **kwargs):
        super(DummyBody, self).__init__()

        self.output_size = input_size

    def forward(self, x):
        return x
