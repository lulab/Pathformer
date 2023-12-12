import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support,classification_report
import scanpy as sc
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
import sys
import argparse, sys, os, errno
import random
# sys.path.append(r'/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/MOGONet/')
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from models import init_model_dict, init_optim


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, stop=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.save_epoch = True
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        if len(monitor) == 1:
            score = monitor[0]
        else:
            score = np.mean(monitor)
        if self.verbose:
            print(f'epoch: {epoch}')
            print(f'Score: {score}')
        if self.best_epoch is None:
            self.best_epoch = epoch
        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0

        if (self.best_score is None) | (epoch == 1):
            self.best_score = score
        elif (score < self.best_score - self.delta):
            self.counter += 1
            # print(f'EarlyStopping epoch: {epoch}')
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_epoch = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_epoch = True
            self.best_epoch = epoch

def prepare_trte_data(view_list,data_path,label_path,dataset,feature_num,data_type):
    num_view=len(view_list)
    label=pd.read_csv(label_path,sep='\t')
    sample_train=label.loc[label['dataset_'+str(dataset)+'_new']=='discovery','sample_id']
    sample_validation=label.loc[label['dataset_'+str(dataset)+'_new']=='validation','sample_id']
    sample_test=label.loc[label['dataset_'+str(dataset)+'_new']=='test','sample_id']
    label=label.set_index('sample_id')
    y_train=np.array(label.loc[sample_train,'y']).astype('int')
    y_validation = np.array(label.loc[sample_validation, 'y']).astype('int')
    y_test = np.array(label.loc[sample_test, 'y']).astype('int')
    y_num=len(set(label['y']))

    data_tr_list = []
    data_val_list = []
    data_te_list = []
    for type in view_list:
        data = pd.read_csv(data_path + str(dataset) + '/' + str(feature_num) + '/data_' + type  + '.txt',sep='\t')
        data = data.fillna(0)
        X_train = np.array(data.loc[:, sample_train]).T
        scaler = MinMaxScaler().fit(X_train)
        X_train=scaler.transform(X_train)
        X_validation=np.array(data.loc[:,sample_validation]).T
        X_validation=scaler.transform(X_validation)
        X_test=np.array(data.loc[:,sample_test]).T
        X_test=scaler.transform(X_test)
        data_tr_list.append(X_train)
        data_val_list.append(X_validation)
        data_te_list.append(X_test)
    num_tr = data_tr_list[0].shape[0]
    num_val = data_val_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["val"] = list(range(num_tr, (num_tr+num_val)))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    label_te = np.concatenate((y_train, y_test))
    label_val = np.concatenate((y_train, y_validation))

    data_tensor_tr_list=[]
    for i in range(num_view):
        data_tensor_tr_list.append(torch.FloatTensor(data_tr_list[i]))

    data_tensor_val_list = []
    for i in range(num_view):
        data_tensor_val_list.append(torch.FloatTensor(np.concatenate((data_tr_list[i], data_val_list[i]), axis=0)))

    data_tensor_te_list=[]
    for i in range(num_view):
        data_tensor_te_list.append(torch.FloatTensor(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0)))

    return data_tensor_tr_list,data_tensor_val_list,data_tensor_te_list, idx_dict,label_val, label_te,y_num

def gen_trte_adj_mat(data_tensor_tr_list,data_tensor_val_list, data_tensor_te_list, idx_dict, adj_parameter):
    #样本邻阶矩阵
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_val_list = []
    adj_test_list = []
    for i in range(len(data_tensor_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tensor_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tensor_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_val_list.append(gen_test_adj_mat_tensor(data_tensor_val_list[i], idx_dict['tr'], idx_dict['val'], adj_parameter_adaptive,adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_tensor_te_list[i], idx_dict['tr'],idx_dict['te'], adj_parameter_adaptive, adj_metric))

    return adj_train_list,adj_val_list, adj_test_list

def train_epoch(data_list, adj_list, label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none') #交叉熵损失函数
    for m in model_dict:
        model_dict[m].train() #让model变成训练模式,Batch Normalization 和 Dropout 方法模式不同
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()#梯度置零，将梯度初始化为零
        # ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])) #forward结构
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight)) #损失函数
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        # c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
        prob = F.softmax(c, dim=1).data.cpu().numpy()
        return prob,loss_dict
    else:
        return loss_dict

def test_epoch(data_list,adj_list,label,sample_weight,te_idx, model_dict):
    criterion = torch.nn.CrossEntropyLoss(reduction='none') #交叉熵损失函数
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
    loss = c_loss.detach().cpu().numpy().item()
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob,loss

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
    label=np.array(y)
    pre_label = (pro>0.5).astype(int)
    auc=roc_auc_score(label, np.array(pro))
    acc = accuracy_score(label, pre_label)
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc,f1_weighted, f1_macro



def main_model(data_path,label_path,save_path,dataset,feature_num,stop,data_type,num):
    setup_seed(2022)
    view_list= ['count', 'CNV', 'methylation']
    num_epoch_pretrain = 200
    num_epoch = num
    adj_parameter = 2
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    dim_he_list =[100,100,100,100,100,100,100]
    num_view = len(view_list)
    val_inverval = 1

    #数据读取和处理
    #数据读取(data_tr_lis为多组学训练数据，data_trte_list为多组学全部数据，数据index，数据label
    data_tensor_tr_list,data_tensor_val_list,data_tensor_te_list, idx_dict,label_val, label_te,y_num = prepare_trte_data(view_list,data_path,label_path,dataset,feature_num,data_type)
    num_class=len(set(label_val))
    dim_hvcdn = pow(num_class, num_view)
    #train
    labels_tr_tensor = torch.LongTensor(label_te[idx_dict["tr"]])
    sample_weight_tr = cal_sample_weight(label_te[idx_dict["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    #validation
    labels_val_tensor = torch.LongTensor(label_val[idx_dict["val"]])
    sample_weight_val = cal_sample_weight(label_val[idx_dict["val"]], num_class)
    sample_weight_val = torch.FloatTensor(sample_weight_val)
    #test
    labels_te_tensor = torch.LongTensor(label_te[idx_dict["te"]])
    sample_weight_te = cal_sample_weight(label_te[idx_dict["te"]], num_class)
    sample_weight_te = torch.FloatTensor(sample_weight_te)

    adj_tr_list,adj_val_list, adj_te_list = gen_trte_adj_mat(data_tensor_tr_list,data_tensor_val_list, data_tensor_te_list, idx_dict, adj_parameter)#构造邻阶矩阵
    dim_list = [x.shape[1] for x in data_tensor_tr_list]

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)

    #预训练GCN
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        loss_dict=train_epoch(data_tensor_tr_list, adj_tr_list, labels_tr_tensor,
                            sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
        # print("epoch="+str(epoch))
        # print(loss_dict)
    print("\nPretrain GCNs,ok")


    #训练
    print("\nTraining&Test...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    patience = 10
    early_stopping = EarlyStopping(patience=10, verbose=False, delta=1e-2, stop=stop)

    if y_num==2:
        f = open(save_path + 'result.txt', 'w')
        for epoch in range(1,num_epoch+1):
            ######模型训练########
            tr_prob,loss_dict=train_epoch(data_tensor_tr_list, adj_tr_list, labels_tr_tensor,sample_weight_tr, model_dict, optim_dict)
            ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(labels_tr_tensor.data.cpu().numpy(),tr_prob[:, 1])
            # 输出
            epoch_loss=loss_dict['C']
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

            ######模型测试########
            val_prob, val_loss = test_epoch(data_tensor_val_list, adj_val_list, labels_val_tensor, sample_weight_val,idx_dict["val"], model_dict)
            ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(labels_val_tensor.data.cpu().numpy(),val_prob[:, 1])
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
            if epoch % val_inverval == 0:
                te_prob, test_loss = test_epoch(data_tensor_te_list, adj_te_list, labels_te_tensor, sample_weight_te,idx_dict["te"], model_dict)
                ACC_test, AUC_test, f1_weighted_test, f1_macro_test = get_roc(labels_te_tensor.data.cpu().numpy(),te_prob[:, 1])
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

            if (early_stopping.early_stop)&(epoch>=stop):
                te_prob, test_loss = test_epoch(data_tensor_te_list, adj_te_list, labels_te_tensor, sample_weight_te,idx_dict["te"], model_dict)
                ACC_test, AUC_test, f1_weighted_test, f1_macro_test = get_roc(labels_te_tensor.data.cpu().numpy(),te_prob[:, 1])
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
        f.close()
    else:
        f = open(save_path + 'result.txt', 'w')
        for epoch in range(1, num_epoch + 1):
            ######模型训练########
            tr_prob, loss_dict = train_epoch(data_tensor_tr_list, adj_tr_list, labels_tr_tensor,sample_weight_tr, model_dict, optim_dict)
            epoch_loss = loss_dict['C']
            acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = get_roc_multi(labels_tr_tensor.data.cpu().numpy(), tr_prob)
            y_train_new = np.array(labels_tr_tensor.data.cpu().numpy()).copy()
            y_train_new[y_train_new >= 1] = 1
            ACC_train_2, AUC_train_2, f1_weighted_train_2, f1_macro_train_2 = get_roc(np.array(y_train_new),1 - np.array(tr_prob)[:, 0])

            #输出
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


            ######模型测试########
            val_prob, val_loss = test_epoch(data_tensor_val_list, adj_val_list, labels_val_tensor,sample_weight_val,idx_dict["val"], model_dict)
            acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(labels_val_tensor.data.cpu().numpy(), val_prob)
            y_val_new = np.array(labels_val_tensor.data.cpu().numpy()).copy()
            y_val_new[y_val_new >= 1] = 1
            ACC_val_2, AUC_val_2, f1_weighted_val_2, f1_macro_val_2 = get_roc(np.array(y_val_new),1 - np.array(val_prob)[:, 0])
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

            if epoch % val_inverval == 0:
                te_prob, test_loss = test_epoch(data_tensor_te_list, adj_te_list, labels_te_tensor, sample_weight_te,idx_dict["te"], model_dict)
                acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = get_roc_multi(labels_te_tensor.data.cpu().numpy(), te_prob)
                y_test_new = np.array(labels_te_tensor.data.cpu().numpy()).copy()
                y_test_new[y_test_new >= 1] = 1
                ACC_test_2, AUC_test_2, f1_weighted_test_2, f1_macro_test_2 = get_roc(np.array(y_test_new),1 - np.array(te_prob)[:, 0])
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

            if (early_stopping.early_stop) & (epoch >= stop):
                te_prob, test_loss = test_epoch(data_tensor_te_list, adj_te_list, labels_te_tensor, sample_weight_te,idx_dict["te"], model_dict)
                acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = get_roc_multi(labels_te_tensor.data.cpu().numpy(), te_prob)
                y_test_new = np.array(labels_te_tensor.data.cpu().numpy()).copy()
                y_test_new[y_test_new >= 1] = 1
                ACC_test_2, AUC_test_2, f1_weighted_test_2, f1_macro_test_2 = get_roc(np.array(y_test_new),1 - np.array(te_prob)[:, 0])
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
        f.close()

def main(args):
    main_model(args.data_path,args.label_path,args.save_path,args.dataset,args.feature_num,args.stop,args.data_type,args.num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--data_path', type=str,required=True,
                        help='data_path',dest='data_path')
    parser.add_argument('--label_path', type=str,required=True,
                        help='label_path',dest='label_path')
    parser.add_argument('--data_type', type=str, required=True,
                        help='data_type', dest='data_type')
    parser.add_argument('--dataset', type=int,required=True,
                        help='dataset',dest='dataset')
    parser.add_argument('--feature_num', type=int,required=True,
                        help='feature_num',dest='feature_num')
    parser.add_argument('--stop', type=int,required=True,
                        help='stop',dest='stop')
    parser.add_argument('--save_path', type=str,required=True,
                        help='save_path',dest='save_path')
    parser.add_argument('--num', type=int, required=True,
                        help='num', dest='num')
    args = parser.parse_args()
    main(args)
