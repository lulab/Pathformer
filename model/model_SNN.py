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
                 attn_dropout,
                 ff_dropout,
                 classifier_dropout,
                 norm_tf):
        super(omics_model, self).__init__()
        self.pathway_model = CustomizedLinear(mask_raw, bias=None)
        self.classifier_model = classifier(classifier_input, classifier_dim, label_dim, classifier_dropout)

    def forward(self,x,m_row):
        m = self.pathway_model(m_row)
        dec_logits = self.classifier_model(m)

        return dec_logits
