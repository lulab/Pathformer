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