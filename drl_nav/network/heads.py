from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drl_nav.utils.image_utils import (
    rgb_to_hsv, is_yellow, is_blue, to_np_image
)


class QNet(nn.Module):
    """
    Q-network. 
    Deep network that learn the relationship between the states and the action values.
    """
    
    def __init__(self, body, action_size):
        super(QNet, self).__init__()

        self.body = body
        self.action_size = action_size
        self.hidden_layers_dim = [2888, 128, 64]
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(size_in, size_out) for size_in, size_out in
            zip(self.hidden_layers_dim[:-1],  self.hidden_layers_dim[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layers_dim[-1], action_size)

        self.optimizer = torch.optim.Adam(self.parameters()) # set the learning rate
        
    def forward(self, x):
        
        x = self.body(x)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        output = self.output_layer(x)
            
        return output

class AuxNet(nn.Module):
    """
    Network that will help training the convolutional body by predicting the
    presence of bananas in each direction (left, front, right).
    """
    def __init__(self, body, n_directions=3, n_colors=2):
        super(AuxNet, self).__init__()
        self.body = body
        self.hidden_layers_dim = [2888, 128]

        self.hidden_layers = nn.ModuleList([
            nn.Linear(size_in, size_out) for size_in, size_out in
            zip(self.hidden_layers_dim[:-1],  self.hidden_layers_dim[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layers_dim[-1], n_directions * n_colors)

        self.optimizer = torch.optim.Adam(self.parameters()) # set the learning rate

    def forward(self, state):

        x = self.body(state)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        output = torch.sigmoid(self.output_layer(x))

        return output

class LabelizerNet(nn.Module):
    """
    Network that help the vision part of the DQN network training
    by giving it banana labels.
    """
    def __init__(self, n_pannels=3) -> None:
        super(LabelizerNet, self).__init__()
        self.n_pannels = n_pannels # left middle right
        self.avg_pool = nn.AvgPool2d(kernel_size=6, stride=2)        
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO process multiple image at once.
        """
        Parse the input state to detect directionally near bananas.
        
        Args:
            state (numpy nd array): state 1*84*84*3 representation
        
        Returns:
            np.ndarray: labels vector for yellow banana detection by pannel.
            np.ndarray: labels vector for blue banana detection by pannel.
        """
        # The intent of average pooling is to blur the image to only detect near bananas.
        output = self.avg_pool(state)
        output_np = to_np_image(output)
        output_hsv = np.apply_along_axis(rgb_to_hsv, 2, output_np)
        
        mask_bananas = (
            np.apply_along_axis(is_yellow, 2, output_hsv).astype(int) - 
            np.apply_along_axis(is_blue, 2, output_hsv).astype(int)
        )
        
        labels_blue = []
        labels_yellow = []
        boundaries = [output_hsv.shape[1] // self.n_pannels * i for i in range(self.n_pannels + 1)]
        for lower_bound, higher_bound in zip(boundaries[:-1], boundaries[1:]):
            
            pannel_mask = mask_bananas[:, lower_bound: higher_bound] 
            
            label_blue_pannel = float(np.min(pannel_mask) == -1) # np min would be sufficient.
            labels_blue.append(label_blue_pannel)
            
            label_yellow_pannel = float(np.max(pannel_mask) == 1)
            labels_yellow.append(label_yellow_pannel)
        
        labels_banana = np.concatenate((labels_yellow, labels_blue))
        labels_banana = torch.from_numpy(labels_banana)
        
        return labels_banana