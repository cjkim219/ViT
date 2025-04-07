import torch
import torch.nn as nn
import torch.nn.functional as F

import config as con


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
    
        self.LayerNorm = nn.LayerNorm(embed_dim)
        
        # self.MultiheadAttention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.MHA = MY_MultiheadAttention(embed_dim, num_heads)
        
        # MLP embedding dim?
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*con.MLP_scale),
            nn.GELU(),
            nn.Dropout(con.Dropout_rate),
            nn.Linear(embed_dim*con.MLP_scale, embed_dim),
            nn.Dropout(con.Dropout_rate)
        )
        
    
    def forward(self, x):
    
        # Multihead Attention
        x = x + self.MHA(self.LayerNorm(x))
        x = x + self.MLP(self.LayerNorm(x))
        
        return x
        
        
        
class MY_MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MY_MultiheadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.proj = nn.Linear(embed_dim, embed_dim*3) 
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, x):
        qkv = self.proj(x).reshape(x.size(0), x.size(1), 3, self.num_heads, self.embed_dim//self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim = -1)
        
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        
        return self.out_proj(out)