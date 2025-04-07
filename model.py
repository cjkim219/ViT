import torch
import torch.nn as nn
import torch.nn.functional as F

import config as con

from utils import *
from transformer import *


class ViT(nn.Module):
  def __init__(self, Hidden_dim, num_heads, batch_size, img_size):
    super(ViT, self).__init__()
    
    self.Hidden_dim = Hidden_dim
    self.num_heads = num_heads
    self.batch_size = batch_size

    ##############
    
    self.LinearProjection = nn.Linear(con.flatten_size, Hidden_dim)
    
    self.clsss_token = nn.Parameter(torch.randn(1, Hidden_dim))
    
    self.pos_embedding = nn.Parameter(torch.randn((img_size // con.patch_size)**2 + 1, Hidden_dim))
    
    self.transformerencoder = nn.Sequential()
    for i in range(con.Encoder_iter):
      self.transformerencoder.add_module(f"{i}-th encoder", TransformerEncoder(Hidden_dim, num_heads))
    
    self.MLPHead = nn.Sequential(
        # nn.Linear(Hidden_dim, Hidden_dim*con.MLP_scale),
        # nn.Dropout(con.Dropout_rate),
        # nn.Tanh(),
        # nn.Linear(Hidden_dim*con.MLP_scale, con.num_classes),
        # nn.Dropout(con.Dropout_rate)
        nn.Linear(Hidden_dim, con.num_classes)
    )
    
    self.LayerNorm = nn.LayerNorm(Hidden_dim)
    
    ##############
        
    
  def forward(self, x):
    
    B, _, _, _ = x.size()
      
    x = image_split(x, con.patch_size)
    x = self.LinearProjection(x)
    
    class_token = self.clsss_token.repeat(B, 1, 1)
    PE = self.pos_embedding.repeat(B, 1, 1)
    
    x = torch.cat([class_token, x], dim = 1) + PE
   
    x = self.transformerencoder(x)
    
    return self.MLPHead(self.LayerNorm(x[:, 0]))