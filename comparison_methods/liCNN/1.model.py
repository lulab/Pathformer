import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, \
    classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scanpy as sc
import anndata as ad
from scipy.stats import rankdata
import sys


######################
#######函数############
######################
class lfCNN(torch.nn.Module):
    def __init__(self,in_dim,num_classes):
        super(lfCNN, self).__init__()
        self.FC_1 = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
                    nn.ReLU(),
                    nn.MaxPool1d(100),
                    nn.Flatten())
        self.FC_2 = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
                    nn.ReLU(),
                    nn.MaxPool1d(100),
                    nn.Flatten())
        self.FC_3 = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
                    nn.ReLU(),
                    nn.MaxPool1d(100),
                    nn.Flatten())
        self.FC_merge = nn.Sequential(
                    nn.Linear(in_features=int((in_dim[0]-300+1)/100)*32+int((in_dim[1]-300+1)/100)*32+int((in_dim[2]-300+1)/100)*32, out_features=100),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(in_features=100, out_features=50),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(in_features=50, out_features=10),
                    nn.ReLU(),
                    nn.Linear(in_features=10, out_features=num_classes))
        self.softmax = nn.Softmax(dim=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,in_dim,num_classes):
        x_1=self.FC_1(x[:,:in_dim[0]].unsqueeze(2).permute(0,2,1))
        x_2 = self.FC_2(x[:,in_dim[0]+1:in_dim[0]+in_dim[1]+1].unsqueeze(2).permute(0,2,1))
        x_3 = self.FC_3(x[:,in_dim[0]+in_dim[1]:in_dim[0]+in_dim[1]+in_dim[2]+1].unsqueeze(2).permute(0,2,1))
        x=torch.cat((x_1,x_2,x_3),1)
        x=self.FC_merge(x)
        if num_classes==2:
            dec_logits=self.sigmoid(x)
        else:
            dec_logits = self.softmax(x)
        return dec_logits

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


def save_ckpt(epoch, model, optimizer, losses, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'losses': losses, },
        f'{ckpt_folder}{model_name}_{epoch}.pth')


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

def main_model(label_path, data_path, dataset, save_path, epoch_num, learning_rate, patience, delta,stop_epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = 2022  # Random seed.
    setup_seed(seed)
    # epoch_num = 500
    batch_size = 32  # Number of batch size.
    learning_rate = learning_rate # Learning rate.
    grad_acc = 1  # Number of gradient accumulation.梯度累积
    valid_every = 1  # Number of training epochs between twice validation.

    SEED = seed
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    GRADIENT_ACCUMULATION = grad_acc
    VALIDATE_EVERY = valid_every


    ######################
    #######main############
    ######################

    ####数据读入####
    label = pd.read_csv(label_path, sep='\t')
    num_y = len(set(label['y']))
    train_sample = list(label.loc[label['dataset_' + str(dataset) + '_new'] == 'train', 'sample_id'])
    validation_sample = list(label.loc[label['dataset_' + str(dataset) + '_new'] == 'validation', 'sample_id'])
    test_sample = list(label.loc[label['dataset_' + str(dataset) + '_new'] == 'test', 'sample_id'])

    label = label.set_index('sample_id')

    train_label = label.loc[train_sample, ['y']].values
    train_label = train_label.astype(int)
    validation_label = label.loc[validation_sample, ['y']].values
    validation_label = validation_label.astype(int)
    test_label = label.loc[test_sample, ['y']].values
    test_label = test_label.astype(int)
    label_dict, train_label_ = np.unique(train_label,return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    class_num = np.unique(train_label, return_counts=True)[1].tolist()
    class_weight = torch.tensor([sum(class_num) / (2 * x) for x in class_num])

    train_label = torch.Tensor(train_label).long()
    validation_label = torch.Tensor(validation_label).long()
    test_label = torch.Tensor(test_label).long()


    data_count = pd.read_csv(data_path+'/data_count.txt', sep='\t')
    data_CNV = pd.read_csv(data_path + '/data_CNV.txt', sep='\t')
    data_methylation = pd.read_csv(data_path + '/data_methylation.txt', sep='\t')
    data_count = data_count.set_index(data_count.columns[0])
    data_CNV = data_CNV.set_index(data_CNV.columns[0])
    data_methylation = data_methylation.set_index(data_methylation.columns[0])

    data_count_train=np.array(data_count[train_sample]).T
    data_count_validation = np.array(data_count[validation_sample]).T
    data_count_test = np.array(data_count[test_sample]).T

    data_CNV_train=np.array(data_CNV[train_sample]).T
    data_CNV_validation = np.array(data_CNV[validation_sample]).T
    data_CNV_test = np.array(data_CNV[test_sample]).T

    data_methylation_train=np.array(data_methylation[train_sample]).T
    data_methylation_validation = np.array(data_methylation[validation_sample]).T
    data_methylation_test = np.array(data_methylation[test_sample]).T

    
    count_feature_len=data_count_train.shape[1]
    CNV_feature_len = data_CNV_train.shape[1]
    methylation_feature_len = data_methylation_train.shape[1]
    data_train = np.concatenate((data_count_train,data_CNV_train,data_methylation_train),axis=1)
    data_validation = np.concatenate((data_count_validation, data_CNV_validation, data_methylation_validation), axis=1)
    data_test = np.concatenate((data_count_test, data_CNV_test, data_methylation_test), axis=1)

    train_dataset = SCDataset(data_train, train_label)
    val_dataset = SCDataset(data_validation, validation_label)
    test_dataset = SCDataset(data_test, test_label)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = lfCNN(in_dim=[count_feature_len,CNV_feature_len,methylation_feature_len],num_classes=num_y)

    ####模型训练#####
    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta,stop=stop_epoch)


    if num_y == 2:
        f = open(save_path + 'result.txt', 'w')
        for epoch in range(1, epoch_num):
            ######模型训练########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dict.shape[0]])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 10 == 1:
                    print(index)
                data = data
                labels = labels
                labels = labels[:, 0]
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(data,in_dim=[count_feature_len,CNV_feature_len,methylation_feature_len],num_classes=num_y)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(data,in_dim=[count_feature_len,CNV_feature_len,methylation_feature_len],num_classes=num_y)
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
            ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(np.array(y_train),np.array(predict_train)[:, 1])
            # 输出
            f.write(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}' + '\n')
            f.write('ACC_train:' + str(ACC_train) + '\n')
            f.write('auc_train:' + str(AUC_train) + '\n')
            f.write('f1_weighted_train:' + str(f1_weighted_train) + '\n')
            f.write('f1_macro_train:' + str(f1_macro_train) + '\n')

            print(f' ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
            print('ACC_train:', ACC_train)
            print('auc_train:', AUC_train)
            print('f1_weighted_train:', f1_weighted_train)
            print('f1_macro_train:', f1_macro_train)

            ######模型验证########
            model.eval()
            running_loss = 0.0
            y_val = []
            predict_val = np.zeros([len(validation_sample), label_dict.shape[0]])
            with torch.no_grad():
                for index, (data, labels) in enumerate(val_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data
                    labels = labels
                    labels = labels[:, 0]
                    logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                   num_classes=num_y)
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
            early_stopping([f1_macro_val], epoch)
            ######模型测试#########
            if epoch % VALIDATE_EVERY == 0:
                model.eval()
                running_loss = 0.0
                y_test = []
                predict_test = np.zeros([len(test_sample), label_dict.shape[0]])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(test_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data
                        labels = labels
                        labels = labels[:, 0]
                        logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                       num_classes=num_y)
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
            ######模型最终测试#########
            if (early_stopping.early_stop) & (epoch > stop_epoch):
                model.eval()
                running_loss = 0.0
                y_test = []
                predict_test = np.zeros([len(test_sample), label_dict.shape[0]])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(test_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data
                        labels = labels
                        labels = labels[:, 0]
                        logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                       num_classes=num_y)
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
                break
        # save_ckpt(epoch, model, optimizer, epoch_loss, model_name, save_path + '/ckpt/')
        f.close()
    else:
        f = open(save_path + 'result.txt', 'w')
        for epoch in range(1, epoch_num):
            ######模型训练########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dict.shape[0]])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 10 == 1:
                    print(index)
                data = data
                labels = labels
                labels = labels[:, 0]
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(data,in_dim=[count_feature_len,CNV_feature_len,methylation_feature_len],num_classes=num_y)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(data,in_dim=[count_feature_len,CNV_feature_len,methylation_feature_len],num_classes=num_y)
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

            ######模型验证########
            model.eval()
            running_loss = 0.0
            y_val = []
            predict_val = np.zeros([len(validation_sample), label_dict.shape[0]])
            with torch.no_grad():
                for index, (data, labels) in enumerate(val_loader):
                    index += 1
                    if index % 100 == 1:
                        print(index)
                    data = data
                    labels = labels
                    labels = labels[:, 0]
                    logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                   num_classes=num_y)
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
            early_stopping([f1_macro_val_2], epoch)
            ######模型测试#########
            if epoch % VALIDATE_EVERY == 0:
                model.eval()
                running_loss = 0.0
                y_test = []
                predict_test = np.zeros([len(test_sample), label_dict.shape[0]])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(test_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data
                        labels = labels
                        labels = labels[:, 0]
                        logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                       num_classes=num_y)
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
            ######模型最终测试#########
            if early_stopping.early_stop:
                model.eval()
                running_loss = 0.0
                y_test = []
                predict_test = np.zeros([len(test_sample), label_dict.shape[0]])
                with torch.no_grad():
                    for index, (data, labels) in enumerate(test_loader):
                        index += 1
                        if index % 100 == 1:
                            print(index)
                        data = data
                        labels = labels
                        labels = labels[:, 0]
                        logits = model(data, in_dim=[count_feature_len, CNV_feature_len, methylation_feature_len],
                                       num_classes=num_y)
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
                break
        # save_ckpt(epoch, model, optimizer, epoch_loss, model_name, save_path + '/ckpt/')
        f.close()

def main(args):

    main_model(args.label_path, args.data_path, args.dataset, args.save_path, args.epoch_num,
               args.learning_rate, args.patience, args.delta,args.stop_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--epoch_num', type=int, required=True,
                        help='epoch_num', dest='epoch_num')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning_rate', dest='learning_rate')
    parser.add_argument('--patience', type=int, required=True,
                        help='patience', dest='patience')
    parser.add_argument('--delta', type=float, required=True,
                        help='delta', dest='delta')
    parser.add_argument('--stop_epoch', type=int, required=True,
                        help='stop_epoch', dest='stop_epoch')
    args = parser.parse_args()
    main(args)
