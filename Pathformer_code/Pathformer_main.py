import sys
import os
import random
import pandas as pd
import numpy as np
from einops import rearrange, repeat, reduce
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD, AdamW,RMSprop
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

from Pathformer import pathformer_model
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_train_test_model(modal_all_path,modal_select_path,gene_all,gene_select,pathway_gene_w,pathway_crosstalk_network,
                          data_path,label_path, save_path,dataset,model_name,model_save,batch_size,gradient_num,epoch_num,
                          early_stopping_type,patience, delta,stop_epoch,test_each_epoch_no,depth,heads,dim_head,beta, attn_dropout,
                          ff_dropout, classifier_dropout, lr_max,lr_min):
    #######################
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_seed(2022)
    BATCH_SIZE = batch_size

    #############################
    #########Data load###########
    #############################
    ###Sample data load
    label = pd.read_csv(label_path, sep='\t')
    train_sample=list(label.loc[label['dataset_' + str(dataset)+'_new'] == 'train', :].index)
    test_sample=list(label.loc[label['dataset_' + str(dataset)+'_new'] == 'test', :].index)
    train_label = label.loc[train_sample, ['y']].values
    train_label = train_label.astype(int)
    test_label = label.loc[test_sample, ['y']].values
    test_label = test_label.astype(int)
    data = np.load(file=data_path)

    gene_all_data = pd.read_csv(gene_all, header=None)
    gene_all_data.columns = ['gene_id']
    gene_all_data['index'] = range(len(gene_all_data))
    gene_all_data = gene_all_data.set_index('gene_id')
    gene_select_data = pd.read_csv(gene_select, header=None)
    gene_select_index = list(gene_all_data.loc[list(gene_select_data[0]), 'index'])

    if (modal_all_path == 'None')|(modal_select_path == 'None'):
        modal_select_index=list(range(data.shape[2]))
    else:
        modal_all_data=pd.read_csv(modal_all_path,header=None)
        modal_all_data.columns=['modal_type']
        modal_all_data['index'] = range(len(modal_all_data))
        modal_all_data = modal_all_data.set_index('modal_type')
        modal_select_data = pd.read_csv(modal_select_path, header=None)
        modal_select_index = list(modal_all_data.loc[list(modal_select_data[0]), 'index'])

    data_train = data[train_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
    data_test = data[test_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]

    train_dataset = SCDataset(data_train, train_label)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    test_dataset = SCDataset(data_test, test_label)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    if (test_each_epoch_no==1):
        validation_sample = list(label.loc[label['dataset_' + str(dataset)] == 'validation', :].index)
        validation_label = label.loc[validation_sample, ['y']].values
        validation_label = validation_label.astype(int)
        data_validation = data[validation_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
        val_dataset = SCDataset(data_validation, validation_label)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    ###Pathway crosstalk netwark load
    gene_pathway = np.load(file=pathway_gene_w)
    gene_pathway = torch.LongTensor(gene_pathway)
    pathway_network = np.load(file=pathway_crosstalk_network)
    pathway_network[np.isnan(pathway_network)] = 0
    pathway_network = torch.Tensor(pathway_network).to(device)

    #######################
    ### Model&optimizer ###
    #######################

    #####hyperparameter
    EPOCH_NUM = epoch_num
    LEARNING_RATE_MAX = lr_max
    LEARNING_RATE_MIN = lr_min
    GRADIENT_ACCUMULATION = gradient_num
    DEPTH = depth #3
    HEAD = heads #8
    HEAD_DIM = dim_head #32
    classifier_dim = [300, 200, 100]

    N_PRJ = data_train.shape[2]  # length of gene embedding
    N_GENE = data_train.shape[1] # number of gene
    col_dim = gene_pathway.shape[1]
    if N_PRJ<=2:
        embeding=True
        embeding_num = 32
        row_dim=embeding_num
        classifier_input = embeding_num * gene_pathway.shape[1]
    else:
        embeding = False
        row_dim = N_PRJ
        classifier_input = N_PRJ * gene_pathway.shape[1]
    mask_raw = gene_pathway
    label_dim = len(set(label['y']))
    class_num = np.unique(np.array(label['y']), return_counts=True)[1].tolist()
    class_weight = torch.tensor([sum(class_num) /(2*x) for x in class_num])


    #####Model
    model = pathformer_model(mask_raw=mask_raw,
                        row_dim=row_dim,
                        col_dim=col_dim,
                        depth=DEPTH,
                        heads=HEAD,
                        dim_head=HEAD_DIM,
                        classifier_input=classifier_input,
                        classifier_dim=classifier_dim,
                        label_dim=label_dim,
                        embeding=embeding,
                        embeding_num=embeding_num,
                        beta=beta,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        classifier_dropout=classifier_dropout).to(device)

    #####optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE_MAX)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,max_lr=LEARNING_RATE_MAX,min_lr=LEARNING_RATE_MIN,
                                              first_cycle_steps=15, cycle_mult=2,warmup_steps=5,gamma=0.9)

    loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta,stop=stop_epoch)

    ##############################################
    ###### Model Training & Validation & Test ####
    ##############################################
    if label_dim == 2:
        f = open(save_path + '/result.txt', 'w')
        for epoch in range(1, EPOCH_NUM+1):
            ###### Model Training ########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dim])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 100 == 1:
                    print(index)
                data = data.to(device)
                labels = labels.to(device)
                labels = labels[:, 0]
                pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
                y_train.extend(labels.tolist())
                if (index) * BATCH_SIZE <= train_label.shape[0]:
                    predict_train[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = dec_logits.data.cpu().numpy()
                else:
                    predict_train[(index - 1) * BATCH_SIZE:, :] = dec_logits.data.cpu().numpy()

                running_loss += loss.item()
            epoch_loss = running_loss / index
            ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(np.array(y_train),np.array(predict_train)[:, 1])
            scheduler.step()
            lr_epoch = scheduler.get_lr()[0]

            # output
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

            ###### Model Validation ########
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
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()
                    y_val.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= len(validation_sample):
                        predict_val[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                    else:
                        predict_val[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
            val_loss = running_loss / index
            ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(np.array(y_val), np.array(predict_val)[:, 1])

            # output
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

            ###### Model Test ########
            if (test_each_epoch_no==1):
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
                        pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_test.extend(labels.tolist())
                        if (index) * BATCH_SIZE <= len(test_sample):
                            predict_test[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                        else:
                            predict_test[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
                test_loss = running_loss / index
                ACC_test, AUC_test, f1_weighted_test, f1_macro_test = get_roc(np.array(y_test),
                                                                              np.array(predict_test)[:, 1])

                # output
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

            ####### early_stopping ######
            if (early_stopping_type=='f1_macro')|(early_stopping_type=='f1_macro_2'):
                early_stopping_score=f1_macro_val
            elif (early_stopping_type=='AUC')|(early_stopping_type=='AUC_2'):
                early_stopping_score = AUC_val

            early_stopping([early_stopping_score], epoch)
            if model_save&(epoch>stop_epoch-5)&(early_stopping.save_epoch) :
                save_ckpt(epoch,model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')
            if (epoch>stop_epoch)&(early_stopping.early_stop) :
                break
        f.close()

    else:
        f = open(save_path + '/result.txt', 'w')
        for epoch in range(1, EPOCH_NUM+1):
            ###### Model Training ########
            model.train()
            running_loss = 0.0
            y_train = []
            predict_train = np.zeros([train_label.shape[0], label_dim])
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % 100 == 1:
                    print(index)
                data = data.to(device)
                labels = labels.to(device)
                labels = labels[:, 0]
                pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                if index % GRADIENT_ACCUMULATION != 0:
                    dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    dec_logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(dec_logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
                y_train.extend(labels.tolist())
                if (index) * BATCH_SIZE <= train_label.shape[0]:
                    predict_train[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = dec_logits.data.cpu().numpy()
                else:
                    predict_train[(index - 1) * BATCH_SIZE:, :] = dec_logits.data.cpu().numpy()

                running_loss += loss.item()
            epoch_loss = running_loss / index
            acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = get_roc_multi(np.array(y_train), predict_train)
            y_train_new = np.array(y_train).copy()
            y_train_new[y_train_new >= 1] = 1
            ACC_train_2, AUC_train_2, f1_weighted_train_2, f1_macro_train_2 = get_roc(np.array(y_train_new),1 - np.array(predict_train)[:, 0])
            scheduler.step()
            lr_epoch = scheduler.get_lr()[0]

            # output
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

            ###### Model Test ########
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
                    pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                    logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()
                    y_val.extend(labels.tolist())
                    if (index) * BATCH_SIZE <= len(validation_sample):
                        predict_val[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                    else:
                        predict_val[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
            val_loss = running_loss / index
            acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(
                np.array(y_val), predict_val)
            y_val_new = np.array(y_val).copy()
            y_val_new[y_val_new >= 1] = 1
            ACC_val_2, AUC_val_2, f1_weighted_val_2, f1_macro_val_2 = get_roc(np.array(y_val_new),
                                                                              1 - np.array(predict_val)[:, 0])

            # output
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

            ###### Model Validation ########
            if (test_each_epoch_no==1):
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
                        pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
                        logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        y_test.extend(labels.tolist())
                        if (index) * BATCH_SIZE <= len(test_sample):
                            predict_test[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
                        else:
                            predict_test[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()
                test_loss = running_loss / index
                acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = get_roc_multi(
                    np.array(y_test), predict_test)
                y_test_new = np.array(y_test).copy()
                y_test_new[y_test_new >= 1] = 1
                ACC_test_2, AUC_test_2, f1_weighted_test_2, f1_macro_test_2 = get_roc(np.array(y_test_new),
                                                                                      1 - np.array(predict_test)[:, 0])

                # output
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

                
            ####### early_stopping ######
            if (early_stopping_type=='f1_macro'):
                early_stopping_score=f1_macro_val
            elif (early_stopping_type=='f1_macro_2'):
                early_stopping_score = f1_macro_val_2
            elif (early_stopping_type=='AUC'):
                early_stopping_score = AUC_val
            elif (early_stopping_type=='AUC_2'):
                early_stopping_score = AUC_val_2

            early_stopping([early_stopping_score], epoch)

            if model_save&(epoch>stop_epoch-5)& (early_stopping.save_epoch):
                save_ckpt(epoch,model, optimizer, scheduler, epoch_loss, model_name, save_path + '/ckpt/')
            if (epoch>stop_epoch)&(early_stopping.early_stop) :
                break
        f.close()


def main(args):

    main_train_test_model(args.modal_all_path,args.modal_select_path,args.gene_all,args.gene_select,args.pathway_gene_w,args.pathway_crosstalk_network,args.data_path,args.label_path, args.save_path,
                   args.dataset,args.model_name,args.model_save,args.batch_size,args.gradient_num,args.epoch_num,args.early_stopping_type,args.patience, args.delta,args.stop_epoch,args.test_each_epoch_no,
                   args.depth,args.heads,args.dim_head,args.beta, args.attn_dropout, args.ff_dropout, args.classifier_dropout, args.lr_max,args.lr_min)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--modal_all_path', type=str, default='None',
                        help='modal_all_path', dest='modal_all_path')
    parser.add_argument('--modal_select_path', type=str, default='None',
                        help='modal_select_path', dest='modal_select_path')
    parser.add_argument('--gene_all', type=str, required=True,
                        help='gene_all', dest='gene_all')
    parser.add_argument('--gene_select', type=str, required=True,
                        help='gene_select', dest='gene_select')
    parser.add_argument('--pathway_gene_w', type=str, required=True,
                        help='pathway_gene_w', dest='pathway_gene_w')
    parser.add_argument('--pathway_crosstalk_network', type=str, required=True,
                        help='pathway_crosstalk_network', dest='pathway_crosstalk_network')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model_name', dest='model_name')
    parser.add_argument('--model_save', type=bool, required=True,
                        help='model_save', dest='model_save')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch_size', dest='batch_size')
    parser.add_argument('--gradient_num', type=int, required=True,
                        help='gradient_num', dest='gradient_num')
    parser.add_argument('--epoch_num', type=int, required=True,
                        help='epoch_num', dest='epoch_num')
    parser.add_argument('--early_stopping_type', type=str, required=True,
                        help='early_stopping_type', dest='early_stopping_type')
    parser.add_argument('--patience', type=int, required=True,
                        help='patience', dest='patience')
    parser.add_argument('--delta', type=float, required=True,
                        help='delta', dest='delta')
    parser.add_argument('--stop_epoch', type=int, required=True,
                        help='stop_epoch', dest='stop_epoch')
    parser.add_argument('--test_each_epoch_no', type=int, required=True,
                        help='test_each_epoch_no', dest='test_each_epoch_no')
    parser.add_argument('--depth', type=int, required=True,
                        help='depth', dest='depth')
    parser.add_argument('--heads', type=int, required=True,
                        help='heads', dest='heads')
    parser.add_argument('--dim_head', type=int, required=True,
                        help='dim_head', dest='dim_head')
    parser.add_argument('--beta', type=float, required=True,
                        help='beta', dest='beta')
    parser.add_argument('--attn_dropout', type=float, required=True,
                        help='attn_dropout', dest='attn_dropout')
    parser.add_argument('--ff_dropout', type=float, required=True,
                        help='ff_dropout', dest='ff_dropout')
    parser.add_argument('--classifier_dropout', type=float, required=True,
                        help='classifier_dropout', dest='classifier_dropout')
    parser.add_argument('--lr_max', type=float, required=True,
                        help='lr_max', dest='lr_max')
    parser.add_argument('--lr_min', type=float, required=True,
                        help='lr_min', dest='lr_min')
    args = parser.parse_args()
    main(args)


