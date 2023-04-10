from typing import Tuple
from math import floor
import numpy as np
import torch


def compute_output_size(body, img_size: int, n_channels: int) -> int:
    """Compute the number of unit on the last flatten layer of a body."""
    for layer in body.children():
        img_size, n_channels = compute_out_img_size(layer, img_size, n_channels)
        
    return img_size **2 * n_channels

def compute_out_img_size(layer, img_size: int, n_channels: int) -> Tuple[int, int]:
    """Count the dimension of the output of a image layer"""
    kernel = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
    stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]

    size_img = floor((img_size - kernel) / stride + 1)
    
    if hasattr(layer, "out_channels"):
        n_channels = layer.out_channels
        
    return size_img, n_channels

def initializer_fc_layers(network):
    """
    Initialize the weight and biais of all layers with a unifom distribution 
    of spread 1 / sqrt(layer_size)
    """
    for layer in network.children():
        
        if hasattr(layer, "children"):
            layer = initializer_fc_layers(layer)
        
        if hasattr(layer, "weight"):
            spread = 1 / np.sqrt(layer.weight.data.size()[0])
            torch.nn.init.uniform_(layer.weight.data, -spread, spread)
            torch.nn.init.uniform_(layer.bias.data, -spread, spread)

    return network