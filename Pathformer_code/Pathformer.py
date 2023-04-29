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
from CustomizedLinear import CustomizedLinear


def exists(val):
    return val is not None

def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class Embedding_layer(nn.Module):
    def __init__(self,feature_len,embeding):
        super(Embedding_layer, self).__init__()
        self.M = torch.nn.Parameter(torch.tensor(scale(np.random.rand(feature_len, embeding))).float(), requires_grad=True)
    def forward(self, enc_inputs):  # X: [batch_size, feature_len,embeding_len]
        X = enc_inputs * self.M
        return X

class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            gating_module=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.gating_module = gating_module
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        if self.gating_module:
            self.gating = nn.Linear(dim, inner_dim)
            nn.init.constant_(self.gating.weight, 0.)
            nn.init.constant_(self.gating.bias, 1.)
        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, attn_bias=None, context=None, context_mask=None, tie_dim=None,output_attentions=False):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        if exists(tie_dim):
            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)
            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        if output_attentions:
            dots_out = dots
        if exists(attn_bias):
            dots = dots + attn_bias
        # attention
        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # gating
        if self.gating_module:
            gates = self.gating(x)
            out = out * gates.sigmoid()
        # combine to out
        out = self.to_out(out)
        if output_attentions:
            return out, attn, dots_out
        else:
            return out


class AxialAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dropout=0.,
            row_attn=True,
            col_attn=True,
            accept_edges=False,
            global_query_attn=False,
            **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        self.heads = heads
        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn
        self.accept_edges = accept_edges

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads,dropout=dropout, **kwargs)

    def forward(self, x, edges=None, output_attentions=False):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w = x.shape

        # axial attention
        if self.row_attn:
            axial_dim = b
            input_fold_eq = 'b h w -> b w h'
            output_fold_eq = 'b h w -> b w h'
        elif self.col_attn:
            axial_dim = b
            input_fold_eq = 'b w h -> b w h'
            output_fold_eq = 'b w h -> b w h'

        x = rearrange(x, input_fold_eq)

        attn_bias = None
        if self.accept_edges and exists(edges):
            attn_bias = repeat(edges, 'b i j-> b x i j', x=self.heads)

        tie_dim = axial_dim if self.global_query_attn else None
        if output_attentions:
            out, attn_out_1, attn_out_2 = self.attn(x, attn_bias=attn_bias, tie_dim=tie_dim,output_attentions=output_attentions)
            out = rearrange(out, output_fold_eq)
            return out, attn_out_1, attn_out_2
        else:
            out = self.attn(x, attn_bias=attn_bias, tie_dim=tie_dim, output_attentions=output_attentions)
            out = rearrange(out, output_fold_eq)
            return out


class OuterMean(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=None,
            eps=1e-5
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        hidden_dim = default(hidden_dim, dim)
        if self.dim > 1:
            self.norm = nn.LayerNorm(dim)
            self.left_proj = nn.Linear(dim, hidden_dim)
            self.right_proj = nn.Linear(dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        if self.dim > 1:
            left = self.left_proj(x)
            right = self.right_proj(x)
            outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')
        else:
            outer = rearrange(x, 'b m i d -> b m i () d') * rearrange(x, 'b m j d -> b m () j d')
        outer = outer.mean(dim=1)
        if self.dim > 1:
            return self.proj_out(outer)
        else:
            return outer


# main evoformer class
class EvoformerBlock(nn.Module):
    def __init__(
            self,
            *,
            row_dim,
            col_dim,
            heads,
            dim_head,
            beta,
            attn_dropout=0.,
            ff_dropout=0.,
    ):
        super().__init__()
        self.beta=beta
        self.layer = nn.ModuleList([
            AxialAttention(dim=row_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, row_attn=True,col_attn=False, accept_edges=True),
            AxialAttention(dim=col_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, row_attn=False,col_attn=True),
            FeedForward(dim=row_dim, dropout=ff_dropout),
            FeedForward(dim=col_dim, dropout=ff_dropout)
        ])
        self.outer_mean = OuterMean(dim=1)  # “外积平均值”块将MSA表示转换为配对表示的更新

    def forward(self, inputs, output_attentions=False):
        x, m = inputs
        msa_attn_row,msa_attn_col, msa_ff_row,msa_ff_col = self.layer
        # msa attention and transition
        if output_attentions:
            m_, attn_out_row_1, attn_out_row_2 = msa_attn_row(m,  edges=x,output_attentions=output_attentions)
            m=m_+m
            m = msa_ff_row(m.permute(0,2,1)) + m.permute(0,2,1)
            m=m.permute(0,2,1)
            m_, attn_out_col_1, attn_out_col_2 = msa_attn_col(m, output_attentions=output_attentions)
            m=self.beta*m_+m
            m = msa_ff_col(m) + m
        else:
            m = msa_attn_row(m, edges=x) + m
            m = msa_ff_row(m.permute(0,2,1)) + m.permute(0,2,1)
            m=m.permute(0,2,1)
            m = self.beta*msa_attn_col(m) + m
            m = msa_ff_col(m) + m

        # 更新x
        m=m.unsqueeze(3)
        x=x.unsqueeze(3)
        x = x + self.outer_mean(m)
        if output_attentions:
            return x[:,:,:,0], m[:,:,:,0], attn_out_row_1, attn_out_row_2, attn_out_col_1, attn_out_col_2
        else:
            return x[:,:,:,0], m[:,:,:,0]


class Evoformer(nn.Module):
    def __init__(
            self,
            depth,
            dim_network,
            row_dim,
            col_dim,
            heads,
            dim_head,
            embeding,
            beta,
            attn_dropout=0.,
            ff_dropout=0.,

    ):
        super().__init__()
        self.depth = depth
        self.dim_network = dim_network
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.beta=beta
        self.embeding = embeding
        if self.embeding:
            self.embedding_layer = Embedding_layer(feature_len=self.col_dim, embeding=self.row_dim)
        self.layers = nn.ModuleList(
            [EvoformerBlock(row_dim=self.row_dim,col_dim=self.col_dim, heads=self.heads, dim_head=self.dim_head,beta=self.beta,
                              attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout) for _ in range(self.depth)])

    def forward(self, x, m, output_attentions=False):
        if self.embeding:
            m=self.embedding_layer(m.permute(0,2,1)).permute(0,2,1)
        attn_out_row_1_list = []
        attn_out_row_2_list = []
        attn_out_col_1_list = []
        attn_out_col_2_list = []
        for layer in self.layers:
            inp = (x, m)
            if output_attentions:
                x, m, attn_out_row_1, attn_out_row_2, attn_out_col_1, attn_out_col_2 = layer(inp,output_attentions=output_attentions)
                attn_out_row_1_list.append(attn_out_row_1)
                attn_out_row_2_list.append(attn_out_row_2)
                attn_out_col_1_list.append(attn_out_col_1)
                attn_out_col_2_list.append(attn_out_col_2)
            else:
                x, m, *_ = layer(inp)

        if output_attentions:
            return x, m,attn_out_row_1_list, attn_out_row_2_list, attn_out_col_1_list, attn_out_col_2_list
        else:
            return x, m

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


class pathformer_model(nn.Module):
    def __init__(self,
                 mask_raw,
                 depth,
                 row_dim,
                 col_dim,
                 heads,
                 dim_head,
                 classifier_input,
                 classifier_dim,
                 label_dim,
                 embeding,
                 beta,
                 attn_dropout,
                 ff_dropout,
                 classifier_dropout):
        super(pathformer_model, self).__init__()
        self.pathway_model = CustomizedLinear(mask_raw, bias=None)
        self.Evoformer_model = Evoformer(depth=depth, dim_network=1, row_dim=row_dim,
                                     col_dim=col_dim, heads=heads,dim_head=dim_head,embeding=embeding,
                                     beta=beta,attn_dropout=attn_dropout,ff_dropout=ff_dropout)
        self.classifier_model = classifier(classifier_input, classifier_dim, label_dim, classifier_dropout)

    def forward(self,x,m_row,output_attentions):
        m = self.pathway_model(m_row)
        if output_attentions:
            x, m, attn_out_row_1_list, attn_out_row_2_list, attn_out_col_1_list, attn_out_col_2_list = self.Evoformer_model(x, m,output_attentions=output_attentions)
        else:
            x, m = self.Evoformer_model(x, m, output_attentions=output_attentions)

        dec_logits = self.classifier_model(m)
        if output_attentions:
            return dec_logits,attn_out_row_1_list, attn_out_row_2_list, attn_out_col_1_list, attn_out_col_2_list,x
        else:
            return dec_logits




