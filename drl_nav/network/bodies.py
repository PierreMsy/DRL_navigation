from torch import nn
import torch.nn.functional as F




class ConvBody(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self):
        super(ConvBody, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, input):

        x = self.mp1(F.relu(self.conv1(input)))

        return x