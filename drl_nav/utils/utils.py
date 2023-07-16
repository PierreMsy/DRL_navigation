import torch
import random
import numpy as np


def seed_all(seed_value):

    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
