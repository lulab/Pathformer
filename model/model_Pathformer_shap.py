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
sys.path.append(r'/qhsky1/liuxiaofan_result/model/model_test/CCNet/')
from CustomizedLinear import CustomizedLinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############################################################
#####################模型模块#################################
#############################################################

###分类层###
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

def exists(val):
    return val is not None

def default(val, default_val):
    return val if val is not None else default_val

# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d

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


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.,
            norm_tf=False,
    ):
        super().__init__()
        self.norm_tf=norm_tf
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        if self.norm_tf:
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

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None,output_attentions=False):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print('q:',q.shape)
        # print('k:', k.shape)
        # print('v:', v.shape)
        # scale
        q = q * self.scale
        # query / key similarities

        if exists(tie_dim):
            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)
            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        if output_attentions:
            dots_out = dots
        # add attention bias, if supplied (for pairwise to msa attention communication)
        if exists(attn_bias):
#             # print('dots',dots.shape)
#             # print('attn_bias', attn_bias.shape)
            dots = dots + attn_bias

        # print('dots:', dots.shape)
        # masking
        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2],device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention
        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # print('attn:', attn.shape)
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
        # print('out:', out.shape)
        if output_attentions:
            return out, attn, dots_out
        else:
            return out


class AxialAttention_2(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dropout=0.,
            row_attn=True,
            col_attn=True,
            accept_edges=False,
            global_query_attn=False,
            norm_tf=False,
            **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        self.heads = heads
        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn
        self.accept_edges = accept_edges
        self.norm_tf = norm_tf

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads,dropout=dropout, **kwargs)

    def forward(self, x, edges=None, mask=None, output_attentions=False):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w = x.shape

        # axial attention
        if self.row_attn:
            axial_dim = b
            input_fold_eq = 'b h w -> b w h'
            output_fold_eq = 'b h w -> b w h'
            if exists(mask):
                mask = (mask.sum(dim=1) >= h)
        elif self.col_attn:
            axial_dim = b
            input_fold_eq = 'b w h -> b w h'
            output_fold_eq = 'b w h -> b w h'
            if exists(mask):
                mask = (mask.sum(dim=2) == 0)

        x = rearrange(x, input_fold_eq)

        attn_bias = None
        if self.accept_edges and exists(edges):
            attn_bias = repeat(edges, 'b i j-> b x i j', x=self.heads)

        # print('attn x:',x.shape)
        tie_dim = axial_dim if self.global_query_attn else None
        if output_attentions:
            out, attn_out_1, attn_out_2 = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim,output_attentions=output_attentions)
            out = rearrange(out, output_fold_eq)
            return out, attn_out_1, attn_out_2
        else:
            out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim, output_attentions=output_attentions)
            out = rearrange(out, output_fold_eq)
            return out


class OuterMean(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=None,
            eps=1e-5,
            norm_tf=False
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.norm_tf=norm_tf
        hidden_dim = default(hidden_dim, dim)
        if self.dim > 1:
            self.norm = nn.LayerNorm(dim)
            self.left_proj = nn.Linear(dim, hidden_dim)
            self.right_proj = nn.Linear(dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        # print('outmean:',x.shape)
        # print(self.dim)
        if self.dim > 1:
            # if self.norm_tf:
            #     x = self.norm(x)
            left = self.left_proj(x)
            right = self.right_proj(x)
            outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')
        else:
            outer = rearrange(x, 'b m i d -> b m i () d') * rearrange(x, 'b m j d -> b m () j d')
        # print('outer：',outer.shape)
        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)
        # print('outer：', outer.shape)
        if self.dim > 1:
            return self.proj_out(outer)
        else:
            return outer


# main evoformer class
class EvoformerBlock_2(nn.Module):
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
            norm_tf=False
    ):
        super().__init__()
        self.beta=beta
        self.layer = nn.ModuleList([
            AxialAttention_2(dim=row_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, row_attn=True,col_attn=False, accept_edges=True, norm_tf=norm_tf),
            AxialAttention_2(dim=col_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, row_attn=False,col_attn=True, norm_tf=norm_tf),
            FeedForward(dim=row_dim, dropout=ff_dropout,norm_tf=norm_tf),
            FeedForward(dim=col_dim, dropout=ff_dropout,norm_tf=norm_tf)
        ])
        self.outer_mean = OuterMean(dim=1,norm_tf=norm_tf)  # “外积平均值”块将MSA表示转换为配对表示的更新

    def forward(self, inputs, output_attentions=False):
        x, m, mask, msa_mask = inputs
        msa_attn_row,msa_attn_col, msa_ff_row,msa_ff_col = self.layer
        # msa attention and transition
        if output_attentions:
            m_, attn_out_row_1, attn_out_row_2 = msa_attn_row(m, mask=msa_mask, edges=x,output_attentions=output_attentions)
            m=m_+m
            m = msa_ff_row(m.permute(0,2,1)) + m.permute(0,2,1)
            m=m.permute(0,2,1)
            m_, attn_out_col_1, attn_out_col_2 = msa_attn_col(m, mask=msa_mask,output_attentions=output_attentions)
            m=self.beta*m_+m
            m = msa_ff_col(m) + m
        else:
            # print('msa_attn_row')
            m = msa_attn_row(m, mask=msa_mask, edges=x) + m
            m = msa_ff_row(m.permute(0,2,1)) + m.permute(0,2,1)
            m=m.permute(0,2,1)
            # print('msa_attn_col')
            m = self.beta*msa_attn_col(m, mask=msa_mask) + m
            m = msa_ff_col(m) + m

        # 更新x
        m=m.unsqueeze(3)
        x=x.unsqueeze(3)
        # print('updata')
        x = x + self.outer_mean(m, mask=msa_mask)
        if output_attentions:
            return x[:,:,:,0], m[:,:,:,0],mask, msa_mask, attn_out_row_1, attn_out_row_2, attn_out_col_1, attn_out_col_2
        else:
            return x[:,:,:,0], m[:,:,:,0], mask, msa_mask


class Evoformer_2(nn.Module):
    def __init__(
            self,
            depth,
            dim_network,
            row_dim,
            col_dim,
            heads,
            dim_head,
            beta,
            attn_dropout=0.,
            ff_dropout=0.,
            norm_tf=False

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
        self.norm_tf = norm_tf
        self.beta=beta
        self.layers = nn.ModuleList(
            [EvoformerBlock_2(row_dim=self.row_dim,col_dim=self.col_dim, heads=self.heads, dim_head=self.dim_head,beta=self.beta,norm_tf=self.norm_tf,
                              attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout) for _ in range(self.depth)])

    def forward(self, x, m, mask=None, msa_mask=None, output_attentions=False):
        attn_out_row_1_list = []
        attn_out_row_2_list = []
        attn_out_col_1_list = []
        attn_out_col_2_list = []
        for layer in self.layers:
            inp = (x, m, mask, msa_mask)
            if output_attentions:
                x, m, mask, msa_mask, attn_out_row_1, attn_out_row_2, attn_out_col_1, attn_out_col_2 = layer(inp,output_attentions=output_attentions)
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


class omics_model(nn.Module):
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
                 beta,
                 attn_dropout,
                 ff_dropout,
                 classifier_dropout,
                 norm_tf):
        super(omics_model, self).__init__()
        self.pathway_model = CustomizedLinear(mask_raw, bias=None)
        self.Evoformer = Evoformer_2(depth=depth, dim_network=1, row_dim=row_dim,
                                     col_dim=col_dim, heads=heads,dim_head=dim_head,beta=beta,
                                     attn_dropout=attn_dropout,ff_dropout=ff_dropout,norm_tf=norm_tf)
        self.classifier_model = classifier(classifier_input, classifier_dim, label_dim, classifier_dropout)

    def forward(self,m_row):
        pathway_mx = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_w_filter_rank.npy')
        pathway_mx[np.isnan(pathway_mx)] = 0
        pathway_mx = torch.Tensor(pathway_mx).to(device)
        x=repeat(pathway_mx, 'i j-> x i j', x=m_row.shape[0])
        output_attentions=False
        m = self.pathway_model(m_row)
        if output_attentions:
            x, m, attn_out_row_1_list, attn_out_row_2_list, attn_out_col_1_list, attn_out_col_2_list = self.Evoformer(x, m,output_attentions=output_attentions)
        else:
            x, m = self.Evoformer(x, m, output_attentions=output_attentions)

        dec_logits = self.classifier_model(m)
        if output_attentions:
            return dec_logits,attn_out_row_1_list, attn_out_row_2_list, attn_out_col_1_list, attn_out_col_2_list,x
        else:
            return dec_logits




