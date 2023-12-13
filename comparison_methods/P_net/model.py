import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
import sys
sys.path.append(r'/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/comparison_methods/P_net/')
from data.reactome import ReactomeNetwork
from model.model_layer import Diagonal, SparseTF


def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())  # 将dict_keys对象转换为列表
    genes = list(np.unique(list(itertools.chain.from_iterable(layer_dict.values()))))
    n_pathways = len(pathways)
    n_genes = len(genes)
    mat = np.zeros((n_genes, n_pathways))  # 翻转矩阵维度，使之与数据框的方向一致
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[g_inds, p_ind] = 1  # 注意矩阵维度的调整
    df = pd.DataFrame(mat, index=genes, columns=pathways)  # 调整索引和列的顺序
    return df


def get_layer_maps(genes, n_levels, direction):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        filtered_map = filtered_map.fillna(0)
        filtering_index = filtered_map.columns
        maps.append(filtered_map)
    return maps


class pnet_Sparselayer(nn.Module):
    def __init__(self, mapp,num_classes=2,dropout_rate=0,kernel_initializer='glorot_uniform', activation='tanh', use_bias=True,
                 bias_initializer='zeros',batch_normal=False):
        super(pnet_Sparselayer, self).__init__()
        self.mapp = mapp
        self.names = self.mapp.index
        self.mapp = self.mapp.values
        self.n_genes, self.n_pathways = self.mapp.shape
        self.num_classes=num_classes
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.batch_normal=batch_normal
        self.bias_initializer = bias_initializer
        self.SparseTF_layer = SparseTF(units=self.n_pathways,input_shape=self.n_genes,map=self.mapp, nonzero_ind=None,
                                       kernel_initializer=self.kernel_initializer, activation=self.activation,
                                       use_bias=self.use_bias, bias_initializer=self.bias_initializer)
        self.dense_layer = nn.Linear(self.n_pathways, self.num_classes)
        self.batch_norm_layer = nn.BatchNorm1d(1)
        self.activation_fn = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        outcome = self.SparseTF_layer(x)
        # print('outcome',outcome.min(),outcome.max())
        decision_outcome = self.dense_layer(outcome)
        # print('decision_outcome',decision_outcome.min(),decision_outcome.max())
        if self.batch_normal:
            decision_outcome = self.batch_norm_layer(decision_outcome)
        decision_outcome = self.activation_fn(decision_outcome)
        # print('decision_outcome',decision_outcome.min(),decision_outcome.max())
        outcome = self.dropout_layer(outcome)
        # print('outcome',outcome.min(),outcome.max())
        return outcome,decision_outcome


class get_pnet(nn.Module):
    def __init__(self, genes, n_features, n_genes, n_hidden_layers, dropout, num_classes,activation=None,direction='root_to_leaf',
                 use_bias=True,kernel_initializer=None, bias_initializer='zeros', batch_normal=False,repeated_outcomes=True):
        super(get_pnet, self).__init__()
        self.genes=genes
        self.n_features = n_features
        self.n_genes = n_genes
        self.n_hidden_layers = n_hidden_layers
        self.direction = direction
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.batch_normal = batch_normal
        self.repeated_outcomes=repeated_outcomes
        self.num_classes=num_classes
        self.layer1 = Diagonal(units=self.n_genes, input_shape=self.n_features, activation=self.activation,
                               use_bias=self.use_bias,
                               kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        self.dense_layer = nn.Linear(self.n_features, self.num_classes)
        self.dense_layer_out = nn.Linear(self.n_genes, self.num_classes)
        self.batch_norm_layer = nn.BatchNorm1d(1)
        self.dropout_layer = nn.Dropout(p=self.dropout[0])
        self.activation_fn = nn.Sigmoid()

        if self.n_hidden_layers > 0:
            maps = get_layer_maps(self.genes, self.n_hidden_layers, self.direction)
            self.layers = nn.ModuleList([pnet_Sparselayer(maps[i],num_classes=self.num_classes,dropout_rate=self.dropout[i+1],kernel_initializer=self.kernel_initializer,
                                                          activation=self.activation, use_bias=self.use_bias,bias_initializer=self.bias_initializer ) for i in range(len(maps[0:-1]))]) #maps[0:-1]

    def forward(self, x):
        outcome = self.layer1(x)
        # print('outcome', torch.nonzero(outcome).shape)
        decision_outcomes = []
        decision_outcome = self.dense_layer(x)
        if self.batch_normal:
            decision_outcome = self.batch_norm_layer(decision_outcome)
        decision_outcome = self.dense_layer_out(outcome)
        # print('decision_outcome', decision_outcome.min(), decision_outcome.max())
        outcome = self.dropout_layer(outcome)
        if self.batch_normal:
            decision_outcome = self.batch_norm_layer(decision_outcome)
        decision_outcome = self.activation_fn(decision_outcome)
        decision_outcomes.append(decision_outcome)
        for layer in self.layers:
            outcome,decision_outcome=layer(outcome)
            decision_outcomes.append(decision_outcome)
        if self.repeated_outcomes:
            outcome = decision_outcomes
        else:
            outcome = decision_outcomes[-1]
        return outcome





# ######参数#########
# x = torch.Tensor(x)
#
# features = cols
# n_features = x.shape[1]
# if hasattr(cols, 'levels'):
#     genes = cols.levels[0]
# else:
#     genes = cols
# n_features = len(features)
# n_genes = len(genes)
# n_hidden_layers = 5  # 5
# activation = 'tanh'
# dropout = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# sparse = True
# batch_normal = False
# use_bias = True
# kernel_initializer = 'glorot_uniform'
# direction = 'root_to_leaf'
# bias_initializer='zeros'
# repeated_outcomes=True
# loss_weights=1.0

