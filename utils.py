import numpy as np

import torch
import torch.nn as nn


def image_split(x, patch_size):
    B, C, H, W = x.size()
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("image size is not divisible by the patch size")
    
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.contiguous().view(B, C, -1, patch_size, patch_size)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.contiguous().view(B, -1, C * patch_size * patch_size)
    return x