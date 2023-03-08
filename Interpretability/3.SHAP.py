import sys
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD, AdamW, RMSprop
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, \
    LeaveOneOut
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import rankdata
from einops import rearrange, repeat, reduce
import argparse
import random
import h5py
import shap

sys.path.append(r'/qhsky1/liuxiaofan_result/model/model_test/')
from model_Pathformer_shap import omics_model
from BERT.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_attn_1(attn_out):
    attn_weights_new = torch.cat(attn_out, dim=1)
    attn_weights_mean = torch.mean(attn_weights_new, dim=1)
    return attn_weights_mean


def get_attn_2(attn_out):
    attn_out_new = []
    for dots in attn_out:
        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn_out_new.append(attn)
    attn_weights_new = torch.cat(attn_out_new, dim=1)
    attn_weights_mean = torch.mean(attn_weights_new, dim=1)
    return attn_weights_mean


dataset = 10
label_path = '/qhsky1/liuxiaofan/Data/TCGA_new/BRCA_subtype/sample_id/sample_cross_subtype_new_new.txt'
save_path = '/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/10/final_save/'
data_path = '/qhsky1/liuxiaofan/Data/TCGA_new/BRCA_subtype/3.merge_data/data_all.npy'
gene_all = '/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/gene_all.txt'
gene_select = '/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/gene_select_filter.txt'
patience = 10
delta = 1e-4
attn_dropout = 0.2
ff_dropout = 0.2
classifier_dropout = 0.3
stop_epoch = 0
epoch_num = 300
lr = 1e-5
lr_min = 1e-8
scale = False
norm_tf = True
train_random = False
clf_weigth = True
seed_sample=11
seed_sample_num=1
beta=0.1
save_num=28
#######################
#####参数##############
#######################
setup_seed(2022)
epoch_num = epoch_num
batch_size = 1
LEARNING_RATE = lr
LEARNING_RATE_MIN = lr_min
GRADIENT_ACCUMULATION = 1
VALIDATE_EVERY = 1


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index]
        full_seq = torch.from_numpy(full_seq).float()
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class SCDataset_random(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start]
        full_seq = torch.from_numpy(full_seq).float()
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


#############################
#########数据读入#############
#############################
###样本读入
label = pd.read_csv(label_path, sep='\t')
train_sample = list(label.loc[label['dataset_' + str(dataset)] == 'train', :].index)
test_sample = list(label.loc[label['dataset_' + str(dataset)] == 'test', :].index)
validation_sample = list(label.loc[label['dataset_' + str(dataset)] == 'validation', :].index)


train_sample = list(train_sample)
test_sample = list(test_sample)
train_label = label.loc[train_sample, ['y']].values
train_label = train_label.astype(int)
test_label = label.loc[test_sample, ['y']].values
test_label = test_label.astype(int)
validation_label = label.loc[validation_sample, ['y']].values
validation_label = validation_label.astype(int)

gene_all_data = pd.read_csv(gene_all, header=None)
gene_all_data.columns = ['gene_id']
gene_all_data['index'] = range(len(gene_all_data))
gene_all_data = gene_all_data.set_index('gene_id')
gene_select_data = pd.read_csv(gene_select, header=None)
gene_select_index = list(gene_all_data.loc[list(gene_select_data[0]), 'index'])

data = np.load(file=data_path)

data = data[train_sample + test_sample + validation_sample, :, :][:, gene_select_index, :]

label_all = np.concatenate([train_label, test_label, validation_label])


data_dataset = SCDataset(data, label_all)
train_loader = DataLoader(data_dataset, batch_size=40, num_workers=0, pin_memory=True, shuffle=True)
test_loader = DataLoader(data_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
# del data

###通路关系读入
gene_pathway = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_gene_w_filter_rank.npy')
gene_pathway = torch.LongTensor(gene_pathway)
pathway_mx = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_w_filter_rank.npy')
pathway_mx[np.isnan(pathway_mx)] = 0
pathway_mx = torch.Tensor(pathway_mx).to(device)

#######################
#####模型及优化#########
#######################
N_PRJ = 14  # 基因初始编码向量长度
N_GENE = gene_pathway.shape[0]
depth = 3
row_dim = N_PRJ
col_dim = gene_pathway.shape[1]
heads = 8
dim_head = 32
classifier_input = N_PRJ * gene_pathway.shape[1]
classifier_dim = [300, 200, 100]
label_dim = len(set(label['y']))
class_num = np.unique(np.array(label['y']), return_counts=True)[1].tolist()
# class_weight = torch.tensor([(1 - (x / sum(class_num))) for x in class_num])
class_weight = torch.tensor([sum(class_num) / (2 * x) for x in class_num])
mask_raw = gene_pathway
if norm_tf == 1:
    norm_tf = True
    print(norm_tf)
else:
    norm_tf = False
    print(norm_tf)
model = omics_model(mask_raw=mask_raw, depth=depth, row_dim=row_dim, col_dim=col_dim,
                    heads=heads, dim_head=dim_head, classifier_input=classifier_input,
                    classifier_dim=classifier_dim, label_dim=label_dim, beta=beta,
                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, classifier_dropout=classifier_dropout,
                    norm_tf=norm_tf).to(device)

ckpt = torch.load(save_path+'/ckpt/BRCA_'+str(save_num)+'.pth',map_location='cuda:0')
model.load_state_dict(ckpt['model_state_dict'])

model = model.to(device)
model.eval()

data_train = next(iter(train_loader))
(input_train, targets_train) = data_train
background = input_train.permute(0, 2, 1).float().to(device)
explainer = shap.GradientExplainer(model, background)

N_SAMPLES=1
shap_1 = h5py.File(save_path+'/shap_gene_1.h5', 'a')
shap_2 = h5py.File(save_path+'/shap_gene_2.h5', 'a')
shap_3 = h5py.File(save_path+'/shap_gene_3.h5', 'a')
shap_4 = h5py.File(save_path+'/shap_gene_4.h5', 'a')
shap_5 = h5py.File(save_path+'/shap_gene_5.h5', 'a')
y_val = []

for batch_index, (data, labels) in enumerate(test_loader):
    print(batch_index)
    batch_index += 1
    if (batch_index > -1) & (batch_index <= 4000):
        data = data.to(device)
        y = labels.to(device)
        y = y[:, 0]
        shap= explainer.shap_values(data.permute(0, 2, 1), nsamples=N_SAMPLES)
        shap_1[str(batch_index)] = shap[0]
        shap_2[str(batch_index)] = shap[1]
        shap_3[str(batch_index)] = shap[2]
        shap_4[str(batch_index)] = shap[3]
        shap_5[str(batch_index)] = shap[4]
        y_val.append(y.tolist())
    else:
        y = labels.to(device)
        y = y[:, 0]
        y_val.append(y.tolist())
        continue

np.save(file=save_path + '/data_label_SHAP.npy', arr=np.array(y_val))
shap_1.close()
shap_2.close()
shap_3.close()
shap_4.close()
shap_5.close()
# N_SAMPLES = 1
# shap_values = explainer.shap_values(input.permute(0, 2, 1)[:, :, :].float().to(device), nsamples=N_SAMPLES)




