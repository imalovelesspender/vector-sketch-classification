"""Independent implementation of Sihui's documented five-feature architecture.

Reference: transformer-baseline@22d4de9, Transformer 5-features.ipynb.
No Sohyun XAI or hierarchical stroke encoder is imported.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def configurations():
    base = dict(width=128, depth=4, heads=4, ff_dim=384, dropout=.1)
    return [dict(base, name='transformer_point3_zero_xy_control', zero_xy=True),
            dict(base, name='transformer_geometry5_base', zero_xy=False),
            dict(base, name='transformer_geometry5_depth6', depth=6, zero_xy=False),
            dict(base, name='transformer_geometry5_width256', width=256, zero_xy=False),
            dict(base, name='transformer_geometry5_heads8', heads=8, zero_xy=False)]


class Block(nn.Module):
    def __init__(self, width, heads, ff_dim, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid):
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        z = F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        x = x + self.drop(self.out(z.transpose(1,2).reshape(b,t,d)))
        return x + self.drop(self.ff(self.norm2(x)))


class Classifier(nn.Module):
    def __init__(self, config, classes=345):
        super().__init__()
        width = config['width']
        if width % config['heads'] or width % 2:
            raise ValueError('Invalid model width/head count')
        self.zero_xy = config['zero_xy']
        self.input = nn.Linear(5,width)
        self.cls = nn.Parameter(torch.empty(1,1,width))
        nn.init.normal_(self.cls,std=.02)
        positions = torch.arange(250,dtype=torch.float32)[:,None]
        freq = torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
        encoding = torch.zeros(250,width)
        encoding[:,0::2],encoding[:,1::2] = torch.sin(positions*freq),torch.cos(positions*freq)
        self.register_buffer('positions',encoding[None,:,:])
        self.blocks = nn.ModuleList([Block(width,config['heads'],config['ff_dim'],config['dropout'])
                                    for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,classes)

    def embed(self, features, lengths):
        if features.ndim != 3 or features.shape[-1] != 5 or not 1 <= features.shape[1] <= 250:
            raise ValueError('Expected [B,T,5], 1 <= T <= 250')
        b,t,_ = features.shape
        lengths = lengths.to(features.device)
        if lengths.shape != (b,) or lengths.dtype not in (torch.int32,torch.int64) or torch.any(lengths<1) or torch.any(lengths>t):
            raise ValueError('Invalid lengths')
        valid = torch.arange(t,device=features.device)[None,:] < lengths[:,None]
        clean = features.masked_fill(~valid[:,:,None],0)
        if self.zero_xy:
            clean = torch.cat([clean[:,:,:3],torch.zeros_like(clean[:,:,3:])],dim=-1)
        x = self.input(clean)
        x = x + self.positions[:,:t].to(x.dtype)
        x = torch.cat([self.cls.expand(b,-1,-1),x],dim=1)
        valid = torch.cat([torch.ones(b,1,dtype=torch.bool,device=x.device),valid],dim=1)
        for block in self.blocks:
            x = block(x,valid)
        return self.norm(x[:,0])

    def forward(self,features,lengths):
        return self.head(self.embed(features,lengths))
