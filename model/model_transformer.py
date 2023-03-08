import math
from math import sqrt
import numpy as np
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from sklearn.preprocessing import scale
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import os
import sys
from torch import nn, einsum
from einops import rearrange
sys.path.append(r'/qhsky1/liuxiaofan_result/model/model_test/CCNet/')
from CustomizedLinear import CustomizedLinear



#############################################################
#####################模型模块#################################
#############################################################


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            # print('attn_x', x.shape)
            x = ff(x)
            # print('ff_x', x.shape)

        return x
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class classifier(nn.Module):
    def __init__(self,classifier_input,classifier_dim,label_dim,dropout_p):
        super(classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=classifier_input, out_features=classifier_dim[0]),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=classifier_dim[0], out_features=classifier_dim[1]),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=classifier_dim[1], out_features=classifier_dim[2]),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=classifier_dim[2], out_features=label_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, enc_outputs):
        dec_logits=self.layer(enc_outputs)
        return dec_logits.view(-1, dec_logits.size(-1))

# main class
class omics_model(nn.Module):
    def __init__(self,
                 mask_raw,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 classifier_input,
                 classifier_dim,
                 label_dim,
                 attn_dropout,
                 ff_dropout,
                 classifier_dropout,
                 mlp_hidden_mults=(4, 2),
                 mlp_act=None):
        super(omics_model, self).__init__()
        self.pathway_model = CustomizedLinear(mask_raw, bias=None)
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout)
        # mlp to logits
        input_size = classifier_input
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, label_dim]

        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.classifier_model = classifier(classifier_input, classifier_dim, label_dim, classifier_dropout)
    def forward(self,x,m_row,output_attentions):
        m = self.pathway_model(m_row)
        m=m.permute(0, 2, 1)
        # print('input',m.shape)
        m=self.transformer(m)
        # flat_m = m.flatten(1)
        # flat_m_new=self.mlp(flat_m)
        # out=nn.Softmax(dim=1)(flat_m_new)
        out=self.classifier_model(m)
        return out




