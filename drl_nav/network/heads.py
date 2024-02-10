from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from drl_nav.utils.image_utils import (
    rgb_to_hsv, is_yellow, is_blue
)
from drl_nav.network.utils import initializer_fc_layers


class QNet(nn.Module):
    """
    Q-network. 
    Deep network that learn the relationship between the states and the action values.
    """
    def __init__(self, body, config, action_size):
        super(QNet, self).__init__()

        self.body = body
        self.action_size = action_size
        self.hidden_layers_dim = [self.body.output_size] + config.hidden_layers
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(size_in, size_out) for size_in, size_out in
            zip(self.hidden_layers_dim[:-1],  self.hidden_layers_dim[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layers_dim[-1], action_size)

        self = initializer_fc_layers(self)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate)
        self.to(config.device)
        
    def forward(self, x) -> torch.tensor:
        """
        Returns:
            torch.rensor [action_size]: action value for all actions. 
        """
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
    def __init__(self, body, config, n_directions=3, n_colors=2):
        super(AuxNet, self).__init__()
        self.body = body
        self.hidden_layers_dim = [self.body.output_size] + [128]

        self.hidden_layers = nn.ModuleList([
            nn.Linear(size_in, size_out) for size_in, size_out in
            zip(self.hidden_layers_dim[:-1],  self.hidden_layers_dim[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layers_dim[-1], n_directions * n_colors)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate_image) # set the learning rate
        self.to(config.device)

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
    
    Stores states until states_buffer_limit is reached and then labelize
    in batch.
    """
    def __init__(self, config, states_buffer_limit:int, n_pannels: int=3) -> None:
        super(LabelizerNet, self).__init__()
        self.states_buffer_limit = states_buffer_limit
        self.n_pannels = n_pannels # left middle right
        self.avg_pool = nn.AvgPool2d(kernel_size=6, stride=2)        
        self.states_to_labelize = []
        self.device = config.device
        self.to(config.device)
    
    def add(self, state: torch.Tensor) -> bool:
        """Add a state to the buffer and retrun if the size limit has been reached"""
        self.states_to_labelize.append(state)
        enough_states = len(self.states_to_labelize) >= self.states_buffer_limit
        return enough_states
            
    def labelize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """labelize all the buffered states an empty the buffer"""
        states = torch.stack(self.states_to_labelize).to(self.device)
        labels_banana = self.forward(states)
        self.states_to_labelize = []

        return states, labels_banana
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parse input states to detect near bananas directionally.

        Args:
            state (torch tensor): Batch * Channels * Height * Width representation
        
        Returns:
            torch tensor: labels vector for colored bananas detection by pannel
                of size Batch * (n_colors * n_pannels).
        """
        if states.dim() == 3:
            # transform single image as a batch of one image.
            states = torch.unsqueeze(states, 0)
        
        pooled_states = self.avg_pool(states)
        np_states = np.moveaxis(pooled_states.numpy(), 1, 3)
        
        states_hsv = np.apply_along_axis(rgb_to_hsv, 3, np_states)
        
        mask_bananas = (
            np.apply_along_axis(is_yellow, 3, states_hsv).astype(int) - 
            np.apply_along_axis(is_blue, 3, states_hsv).astype(int)
        )
        
        labels_blue = []
        labels_yellow = []
        boundaries = [states_hsv.shape[1] // self.n_pannels * i for i in range(self.n_pannels + 1)]
        for lower_bound, higher_bound in zip(boundaries[:-1], boundaries[1:]):
            
            pannel_mask = mask_bananas[:, :, lower_bound: higher_bound] 
            
            label_yellow_pannel = np.max(pannel_mask, axis=(1, 2)) # max on each pannel.
            labels_yellow.append(label_yellow_pannel)
            
            label_blue_pannel = - np.min(pannel_mask, axis=(1, 2)) # min on each pannel.
            labels_blue.append(label_blue_pannel)
            
        labels_yellow = np.stack(labels_yellow, axis=1)
        labels_blue = np.stack(labels_blue, axis=1)
            
        labels_banana = np.concatenate((labels_yellow, labels_blue), axis=1).astype(float)
        labels_banana = torch.from_numpy(labels_banana).to(self.device)
            
        return labels_banana