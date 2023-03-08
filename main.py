import sys
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD, AdamW,RMSprop
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit,RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import rankdata
from einops import rearrange, repeat, reduce
import argparse
import random

sys.path.append(r'/qhsky1/liuxiaofan_result/model/model_test/')
from Preprocessing.data_preprocessing import AnimalData, gene_network_select
from model_Pathformer import omics_model
# from CCNet.OmicsCCnet import omics_model
from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, stop=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        if len(monitor) == 1:
            score = monitor[0]
        else:
            score = np.mean(monitor)
        if self.best_epoch is None:
            self.best_epoch = epoch
        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'EarlyStopping best_epoch: {self.best_epoch}')
        else:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch


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


def plot_roc_multi(prob, label):
    pre_label = prob.argmax(axis=1)
    acc = accuracy_score(label, pre_label)
    auc_macro_ovr = roc_auc_score(label, prob, average='macro', multi_class='ovr')
    auc_macro_ovo = roc_auc_score(label, prob, average='macro', multi_class='ovo')
    auc_weighted_ovr = roc_auc_score(label, prob, average='weighted', multi_class='ovr')
    auc_weighted_ovo = roc_auc_score(label, prob, average='weighted', multi_class='ovo')
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')

    return acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro


def get_roc_multi(y, pro):
    y_index = list(set(y))
    y_index.sort()
    pro_ = pro[:, y_index] / pro[:, y_index].sum(axis=1, keepdims=1)
    pro_[np.isnan(pro_)] = 1 / pro_.shape[1]
    acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro = plot_roc_multi(
        pro_, rankdata(y, method='dense') - 1)
    return acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro


def get_roc(y, pro):
    label = np.array(y)
    pre_label = (pro > 0.5).astype(int)
    auc = roc_auc_score(label, np.array(pro))
    acc = accuracy_score(label, pre_label)
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc, f1_weighted, f1_macro


def save_ckpt(epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'scheduler_state_dict': scheduler.state_dict(),
         'losses': losses, },
        f'{ckpt_folder}{model_name}_{epoch}.pth')


def main_model(label_path, save_path, data_path, dataset, gene_all, gene_select, model_name,
               patience, delta,beta, attn_dropout, ff_dropout, classifier_dropout, norm_tf, stop_epoch, epoch_num, lr,
               lr_min,scale, train_random, clf_weigth,batch_size,gradient_num,save_num):
    #######################
    #####参数##############
    #######################
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_seed(2022)
    epoch_num = epoch_num
    batch_size = batch_size
    LEARNING_RATE = lr
    LEARNING_RATE_MIN = lr_min
    GRADIENT_ACCUMULATION = gradient_num
    VALIDATE_EVERY = 1

    #############################
    #########数据读入#############
    #############################
    ###样本读入
    label = pd.read_csv(label_path, sep='\t')
    discovery_sample= list(label.loc[label['dataset_' + str(dataset)] == 'discovery', :].index)
    validation_sample = list(label.loc[label['dataset_' + str(dataset)] == 'validation', :].index)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    for train_index, test_index in rskf.split(np.array(discovery_sample), label.loc[discovery_sample, ['y']].values):
        train_sample = np.array(discovery_sample)[train_index]
        test_sample = np.array(discovery_sample)[test_index]

    train_sample=list(train_sample)
    test_sample=list(test_sample)
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
    data_train = data[train_sample, :, :][:, gene_select_index, :]
    data_test = data[test_sample, :, :][:, gene_select_index, :]
    data_validation = data[validation_sample, :, :][:, gene_select_index, :]

    # ##归一化
    if scale == 1:
        data_train_scale = data_train.copy()
        data_test_scale = data_test.copy()
        data_validation_scale = data_validation.copy()
        for i in range(data_train.shape[2]):
            scale_clf = MinMaxScaler().fit(data_train[:, :, i])
            data_train_scale[:, :, i] = scale_clf.transform(data_train[:, :, i])
            data_test_scale[:, :, i] = scale_clf.transform(data_test[:, :, i])
            data_validation_scale[:, :, i] = scale_clf.transform(data_validation[:, :, i])

        if train_random == 1:
            train_dataset = SCDataset_random(data_train_scale, train_label)
            print('train_random')
        else:
            train_dataset = SCDataset(data_train_scale, train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        test_dataset = SCDataset(data_test_scale, test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        val_dataset = SCDataset(data_validation_scale, validation_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        print('scale')

    else:
        if train_random == 1:
            train_dataset = SCDataset_random(data_train, train_label)
            print('train_random')
        else:
            train_dataset = SCDataset(data_train, train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        test_dataset = SCDataset(data_test, test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        val_dataset = SCDataset(data_validation, validation_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    ###通路关系读入
    gene_pathway = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_gene_w_filter_rank.npy')
    gene_pathway = torch.LongTensor(gene_pathway)
    pathway_mx = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_w_filter_rank.npy')
    pathway_mx[np.isnan(pathway_mx)] = 0
    pathway_mx = torch.Tensor(pathway_mx).to(device)

    #######################
    #####模型及优化#########
    #######################
    N_PRJ = data_train.shape[2]  # 基因初始编码向量长度
    N_GENE = data_train.shape[1]
    depth = 3
    row_dim = N_PRJ
    col_dim = gene_pathway.shape[1]
    heads = 8
    dim_head = 32
    classifier_input = N_PRJ * gene_pathway.shape[1]
    classifier_dim = [300, 200, 100]
    label_dim = len(set(label['y']))
    class_num = np.unique(np.array(label['y']), return_counts=True)[1].tolist()
    class_weight = torch.tensor([sum(class_num) /(2*x) for x in class_num])
    mask_raw = gene_pathway
    if norm_tf == 1:
        norm_tf = True
        print(norm_tf)
    else:
        norm_tf = False
        print(norm_tf)
    model = omics_model(mask_raw=mask_raw, depth=depth, row_dim=row_dim, col_dim=col_dim,
                        heads=heads, dim_head=dim_head, classifier_input=classifier_input,
                        classifier_dim=classifier_dim, label_dim=label_dim,beta=beta,
                        attn_dropout=attn_dropout, ff_dropout=ff_dropout, classifier_dropout=classifier_dropout,
                        norm_tf=norm_tf).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=LEARNING_RATE_MIN,
        warmup_steps=5,
        gamma=0.9)

    if clf_weigth == 1:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
        print('class_weight')
    else:
        loss_fn = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta,stop=stop_epoch)

    #######################
    ######模型训练及测试####
    #######################
    if label_dim == 2:
        f = open(save_path + '/result.txt', 'w')
        for epoch in range(1, epoch_num+1):
            ######模型训练########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dim])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 10 == 1:
                    print(index)
                data = data.to(device)
                labels = labels.to(device)
                labels = labels[:, 0]
                pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
                y_train.extend(labels.tolist())
                if (index) * batch_size <= train_label.shape[0]:
                    predict_train[(index - 1) * batch_size:(index) * batch_size, :] = dec_logits.data.cpu().numpy()
                else:
                    predict_train[(index - 1) * batch_size:, :] = dec_logits.data.cpu().numpy()

                running_loss += loss.item()
            epoch_loss = running_loss / index
            ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(np.array(y_train),
                                                                              np.array(predict_train)[:, 1])
            scheduler.step()
            lr_epoch = scheduler.get_lr()[0]
            # 输出
            f.write(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}' + '\n')
            f.write('ACC_train:' + str(ACC_train) + '\n')
            f.write('auc_train:' + str(AUC_train) + '\n')
            f.write('f1_weighted_train:' + str(f1_weighted_train) + '\n')
            f.write('f1_macro_train:' + str(f1_macro_train) + '\n')
            f.write('lr:' + str(lr_epoch) + '\n')

            print(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
            print('ACC_train:', ACC_train)
            print('auc_train:', AUC_train)
            print('f1_weighted_train:', f1_weighted_train)
            print('f1_macro_train:', f1_macro_train)
            print('lr:', lr_epoch)

            ######模型测试########
            model.eval()
            running_loss = 0.0
            y_test = []
            predict_test = np.zeros([len(test_sample), label_dim])
            with torch.no_grad():
                for index, (data, labels) in enumerate(test_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data.to(device)
                    labels = labels.to(device)
                    labels = labels[:, 0]
                    pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()
                    y_test.extend(labels.tolist())
                    if (index) * batch_size <= len(test_sample):
                        predict_test[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                    else:
                        predict_test[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
            test_loss = running_loss / index
            ACC_test, AUC_test, f1_weighted_test, f1_macro_test = get_roc(np.array(y_test),
                                                                          np.array(predict_test)[:, 1])

            # 输出
            f.write(f' ==  Epoch: {epoch} | test Loss: {test_loss:.6f}' + '\n')
            f.write('ACC_test:' + str(ACC_test) + '\n')
            f.write('auc_test:' + str(AUC_test) + '\n')
            f.write('f1_weighted_test:' + str(f1_weighted_test) + '\n')
            f.write('f1_macro_test:' + str(f1_macro_test) + '\n')

            print(f' ==  Epoch: {epoch} | test Loss: {test_loss:.6f}')
            print('ACC_test:', ACC_test)
            print('auc_test:', AUC_test)
            print('f1_weighted_test:', f1_weighted_test)
            print('f1_macro_test:', f1_macro_test)
            early_stopping([f1_macro_test], epoch)

            ######模型验证#########
            if epoch % VALIDATE_EVERY == 0:
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device)
                        labels = labels[:, 0]
                        pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * batch_size <= len(validation_sample):
                            predict_val[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(np.array(y_val), np.array(predict_val)[:, 1])
                # 输出
                f.write(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}' + '\n')
                f.write('ACC_val:' + str(ACC_val) + '\n')
                f.write('auc_val:' + str(AUC_val) + '\n')
                f.write('f1_weighted_val:' + str(f1_weighted_val) + '\n')
                f.write('f1_macro_val:' + str(f1_macro_val) + '\n')

                print(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}')
                print('ACC_val:', ACC_val)
                print('auc_val:', AUC_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                if epoch>=save_num:
                    save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')

            if (early_stopping.early_stop) & (epoch > stop_epoch):
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device)
                        labels = labels[:, 0]
                        pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * batch_size <= len(validation_sample):
                            predict_val[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(np.array(y_val), np.array(predict_val)[:, 1])
                # 输出
                f.write(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}' + '\n')
                f.write('ACC_val:' + str(ACC_val) + '\n')
                f.write('auc_val:' + str(AUC_val) + '\n')
                f.write('f1_weighted_val:' + str(f1_weighted_val) + '\n')
                f.write('f1_macro_val:' + str(f1_macro_val) + '\n')
                f.write('best_epoch'+str(early_stopping.best_epoch)+'\n')

                print(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}')
                print('ACC_val:', ACC_val)
                print('auc_val:', AUC_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                if epoch>=save_num:
                    save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')
                break
        f.close()

    else:
        f = open(save_path + '/result.txt', 'w')
        for epoch in range(1, epoch_num+1):
            ######模型训练########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dim])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 10 == 1:
                    print(index)
                data = data.to(device)
                labels = labels.to(device)
                labels = labels[:, 0]
                pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
                y_train.extend(labels.tolist())
                if (index) * batch_size <= train_label.shape[0]:
                    predict_train[(index - 1) * batch_size:(index) * batch_size, :] = dec_logits.data.cpu().numpy()
                else:
                    predict_train[(index - 1) * batch_size:, :] = dec_logits.data.cpu().numpy()

                running_loss += loss.item()
            epoch_loss = running_loss / index
            acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = get_roc_multi(
                np.array(y_train), predict_train)
            y_train_new = np.array(y_train).copy()
            y_train_new[y_train_new >= 1] = 1
            ACC_train_2, AUC_train_2, f1_weighted_train_2, f1_macro_train_2 = get_roc(np.array(y_train_new),
                                                                                      1 - np.array(predict_train)[:, 0])
            scheduler.step()
            lr_epoch = scheduler.get_lr()[0]

            # 输出
            f.write(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}' + '\n')
            f.write('acc_train:' + str(acc_train) + '\n')
            f.write('auc_weighted_ovr_train:' + str(auc_weighted_ovr_train) + '\n')
            f.write('auc_weighted_ovo_train:' + str(auc_weighted_ovo_train) + '\n')
            f.write('auc_macro_ovr_train:' + str(auc_macro_ovr_train) + '\n')
            f.write('auc_macro_ovo_train:' + str(auc_macro_ovo_train) + '\n')
            f.write('f1_weighted_train:' + str(f1_weighted_train) + '\n')
            f.write('f1_macro_train:' + str(f1_macro_train) + '\n')
            f.write('ACC_train_2:' + str(ACC_train_2) + '\n')
            f.write('AUC_train_2:' + str(AUC_train_2) + '\n')
            f.write('f1_weighted_train_2:' + str(f1_weighted_train_2) + '\n')
            f.write('f1_macro_train_2:' + str(f1_macro_train_2) + '\n')
            f.write('lr_epoch:' + str(lr_epoch) + '\n')

            print(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
            print('acc_train:', acc_train)
            print('auc_weighted_ovr_train:', auc_weighted_ovr_train)
            print('auc_weighted_ovo_train:', auc_weighted_ovo_train)
            print('auc_macro_ovr_train:', auc_macro_ovr_train)
            print('auc_macro_ovo_train:', auc_macro_ovo_train)
            print('f1_weighted_train:', f1_weighted_train)
            print('f1_macro_train:', f1_macro_train)
            print('ACC_train_2:', ACC_train_2)
            print('AUC_train_2:', AUC_train_2)
            print('f1_weighted_train_2:', f1_weighted_train_2)
            print('f1_macro_train_2:', f1_macro_train_2)
            print('lr_epoch:', lr_epoch)

            ######模型测试########
            model.eval()
            running_loss = 0.0
            y_test = []
            predict_test = np.zeros([len(test_sample), label_dim])
            with torch.no_grad():
                for index, (data, labels) in enumerate(test_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data.to(device)
                    labels = labels.to(device)
                    labels = labels[:, 0]
                    pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()
                    y_test.extend(labels.tolist())
                    if (index) * batch_size <= len(test_sample):
                        predict_test[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                    else:
                        predict_test[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
            test_loss = running_loss / index
            acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = get_roc_multi(
                np.array(y_test), predict_test)
            y_test_new = np.array(y_test).copy()
            y_test_new[y_test_new >= 1] = 1
            ACC_test_2, AUC_test_2, f1_weighted_test_2, f1_macro_test_2 = get_roc(np.array(y_test_new),
                                                                                  1 - np.array(predict_test)[:, 0])
            # 输出
            f.write(f' ==  Epoch: {epoch} | test Loss: {test_loss:.6f}' + '\n')
            f.write('acc_test:' + str(acc_test) + '\n')
            f.write('auc_weighted_ovr_test:' + str(auc_weighted_ovr_test) + '\n')
            f.write('auc_weighted_ovo_test:' + str(auc_weighted_ovo_test) + '\n')
            f.write('auc_macro_ovr_test:' + str(auc_macro_ovr_test) + '\n')
            f.write('auc_macro_ovo_test:' + str(auc_macro_ovo_test) + '\n')
            f.write('f1_weighted_test:' + str(f1_weighted_test) + '\n')
            f.write('f1_macro_test:' + str(f1_macro_test) + '\n')
            f.write('ACC_test_2:' + str(ACC_test_2) + '\n')
            f.write('AUC_test_2:' + str(AUC_test_2) + '\n')
            f.write('f1_weighted_test_2:' + str(f1_weighted_test_2) + '\n')
            f.write('f1_macro_test_2:' + str(f1_macro_test_2) + '\n')
            print(f' ==  Epoch: {epoch} | test Loss: {test_loss:.6f}')
            print('acc_test:', acc_test)
            print('auc_weighted_ovr_test:', auc_weighted_ovr_test)
            print('auc_weighted_ovo_test:', auc_weighted_ovo_test)
            print('auc_macro_ovr_test:', auc_macro_ovr_test)
            print('auc_macro_ovo_test:', auc_macro_ovo_test)
            print('f1_weighted_test:', f1_weighted_test)
            print('f1_macro_test:', f1_macro_test)
            print('ACC_test_2:', ACC_test_2)
            print('AUC_test_2:', AUC_test_2)
            print('f1_weighted_test_2:', f1_weighted_test_2)
            print('f1_macro_test_2:', f1_macro_test_2)
            early_stopping([f1_macro_test], epoch)

            ######模型验证#########
            if epoch % VALIDATE_EVERY == 0:
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device)
                        labels = labels[:, 0]
                        pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * batch_size <= len(validation_sample):
                            predict_val[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(
                    np.array(y_val), predict_val)
                y_val_new = np.array(y_val).copy()
                y_val_new[y_val_new >= 1] = 1
                ACC_val_2, AUC_val_2, f1_weighted_val_2, f1_macro_val_2 = get_roc(np.array(y_val_new),
                                                                                  1 - np.array(predict_val)[:, 0])
                # 输出
                f.write(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}' + '\n')
                f.write('acc_val:' + str(acc_val) + '\n')
                f.write('auc_weighted_ovr_val:' + str(auc_weighted_ovr_val) + '\n')
                f.write('auc_weighted_ovo_val:' + str(auc_weighted_ovo_val) + '\n')
                f.write('auc_macro_ovr_val:' + str(auc_macro_ovr_val) + '\n')
                f.write('auc_macro_ovo_val:' + str(auc_macro_ovo_val) + '\n')
                f.write('f1_weighted_val:' + str(f1_weighted_val) + '\n')
                f.write('f1_macro_val:' + str(f1_macro_val) + '\n')
                f.write('ACC_val_2:' + str(ACC_val_2) + '\n')
                f.write('AUC_val_2:' + str(AUC_val_2) + '\n')
                f.write('f1_weighted_val_2:' + str(f1_weighted_val_2) + '\n')
                f.write('f1_macro_val_2:' + str(f1_macro_val_2) + '\n')
                print(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}')
                print('acc_val:', acc_val)
                print('auc_weighted_ovr_val:', auc_weighted_ovr_val)
                print('auc_weighted_ovo_val:', auc_weighted_ovo_val)
                print('auc_macro_ovr_val:', auc_macro_ovr_val)
                print('auc_macro_ovo_val:', auc_macro_ovo_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                print('ACC_val_2:', ACC_val_2)
                print('AUC_val_2:', AUC_val_2)
                print('f1_weighted_val_2:', f1_weighted_val_2)
                print('f1_macro_val_2:', f1_macro_val_2)
                if epoch>=save_num:
                    save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')
            if (early_stopping.early_stop) & (epoch > stop_epoch):
                model.eval()
                running_loss = 0.0
                y_val = []
                predict_val = np.zeros([len(validation_sample), label_dim])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data.to(device)
                        labels = labels.to(device)
                        labels = labels[:, 0]
                        pathway_mx_batch = repeat(pathway_mx, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_mx_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_val.extend(labels.tolist())
                        if (index) * batch_size <= len(validation_sample):
                            predict_val[(index - 1) * batch_size:(index) * batch_size, :] = logits.data.cpu().numpy()
                        else:
                            predict_val[(index - 1) * batch_size:, :] = logits.data.cpu().numpy()
                val_loss = running_loss / index
                acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(
                    np.array(y_val), predict_val)
                y_val_new = np.array(y_val).copy()
                y_val_new[y_val_new >= 1] = 1
                ACC_val_2, AUC_val_2, f1_weighted_val_2, f1_macro_val_2 = get_roc(np.array(y_val_new),
                                                                                  1 - np.array(predict_val)[:, 0])
                # 输出
                f.write(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}' + '\n')
                f.write('acc_val:' + str(acc_val) + '\n')
                f.write('auc_weighted_ovr_val:' + str(auc_weighted_ovr_val) + '\n')
                f.write('auc_weighted_ovo_val:' + str(auc_weighted_ovo_val) + '\n')
                f.write('auc_macro_ovr_val:' + str(auc_macro_ovr_val) + '\n')
                f.write('auc_macro_ovo_val:' + str(auc_macro_ovo_val) + '\n')
                f.write('f1_weighted_val:' + str(f1_weighted_val) + '\n')
                f.write('f1_macro_val:' + str(f1_macro_val) + '\n')
                f.write('ACC_val_2:' + str(ACC_val_2) + '\n')
                f.write('AUC_val_2:' + str(AUC_val_2) + '\n')
                f.write('f1_weighted_val_2:' + str(f1_weighted_val_2) + '\n')
                f.write('f1_macro_val_2:' + str(f1_macro_val_2) + '\n')
                print(f' ==  Epoch: {epoch} | val Loss: {val_loss:.6f}')
                print('acc_val:', acc_val)
                print('auc_weighted_ovr_val:', auc_weighted_ovr_val)
                print('auc_weighted_ovo_val:', auc_weighted_ovo_val)
                print('auc_macro_ovr_val:', auc_macro_ovr_val)
                print('auc_macro_ovo_val:', auc_macro_ovo_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                print('ACC_val_2:', ACC_val_2)
                print('AUC_val_2:', AUC_val_2)
                print('f1_weighted_val_2:', f1_weighted_val_2)
                print('f1_macro_val_2:', f1_macro_val_2)
                if epoch>=save_num:
                    save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')
                break
        f.close()


def main(args):

    main_model(args.label_path,args.save_path,args.data_path,args.dataset,args.gene_all,args.gene_select,args.model_name,
               args.patience,args.delta,args.beta,args.attn_dropout,args.ff_dropout,args.classifier_dropout,args.norm_tf,
               args.stop_epoch,args.epoch_num,args.lr,args.lr_min,
               args.scale,args.train_random,args.clf_weigth,args.batch_size,args.gradient_num,args.seed_sample,args.seed_sample_num,args.save_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
    parser.add_argument('--gene_all', type=str, required=True,
                        help='gene_all', dest='gene_all')
    parser.add_argument('--gene_select', type=str, required=True,
                        help='gene_select', dest='gene_select')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model_name', dest='model_name')
    parser.add_argument('--patience', type=int, required=True,
                        help='patience', dest='patience')
    parser.add_argument('--delta', type=float, required=True,
                        help='delta', dest='delta')
    parser.add_argument('--beta', type=float, required=True,
                        help='beta', dest='beta')
    parser.add_argument('--attn_dropout', type=float, required=True,
                        help='attn_dropout', dest='attn_dropout')
    parser.add_argument('--ff_dropout', type=float, required=True,
                        help='ff_dropout', dest='ff_dropout')
    parser.add_argument('--classifier_dropout', type=float, required=True,
                        help='classifier_dropout', dest='classifier_dropout')
    parser.add_argument('--norm_tf', type=int, required=True,
                        help='norm_tf', dest='norm_tf')
    parser.add_argument('--epoch_num', type=int, required=True,
                        help='epoch_num', dest='epoch_num')
    parser.add_argument('--stop_epoch', type=int, required=True,
                        help='stop_epoch', dest='stop_epoch')
    parser.add_argument('--lr', type=float, required=True,
                        help='lr', dest='lr')
    parser.add_argument('--lr_min', type=float, required=True,
                        help='lr_min', dest='lr_min')
    parser.add_argument('--scale', type=int, required=True,
                        help='scale', dest='scale')
    parser.add_argument('--train_random', type=int, required=True,
                        help='train_random', dest='train_random')
    parser.add_argument('--clf_weigth', type=int, required=True,
                        help='clf_weigth', dest='clf_weigth')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch_size', dest='batch_size')
    parser.add_argument('--gradient_num', type=int, required=True,
                        help='gradient_num', dest='gradient_num')
    parser.add_argument('--seed_sample', type=int, required=True,
                        help='seed_sample', dest='seed_sample')
    parser.add_argument('--seed_sample_num', type=int, required=True,
                        help='seed_sample_num', dest='seed_sample_num')
    parser.add_argument('--save_num', type=int, required=True,
                        help='save_num', dest='save_num')
    args = parser.parse_args()
    main(args)





