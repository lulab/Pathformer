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


def main_validation(modal_all_path,modal_select_path,gene_all,gene_select,pathway_gene_w,pathway_crosstalk_network,
                    data_path,label_path,sample_name_path,model_path, save_path,label_dim,evaluate,
                    depth,heads,dim_head,beta, attn_dropout, ff_dropout, classifier_dropout):

    #############################
    #########Data load###########
    #############################
    ###Sample data load
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_seed(2022)
    BATCH_SIZE = 1
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

    data_validation = data[:, :, :][:, gene_select_index, :][:, :, modal_select_index]
    if sample_name_path is None:
        sample_name=np.array(range(data_validation.shape[0]))
    else:
        sample_name=np.array(pd.read_csv(sample_name_path,sep='\t',header=None)[0])

    val_dataset = SCDataset(data_validation,sample_name)
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
    DEPTH = depth  # 3
    HEAD = heads  # 8
    HEAD_DIM = dim_head  # 32
    classifier_dim = [300, 200, 100]

    N_PRJ = data.shape[2]  # length of gene embedding
    N_GENE = data.shape[1]  # number of gene
    col_dim = gene_pathway.shape[1]
    if N_PRJ==1:
        embeding=True
        embeding_num = 32
        row_dim=embeding_num
        classifier_input = embeding_num * gene_pathway.shape[1]
    else:
        embeding = False
        row_dim = N_PRJ
        classifier_input = N_PRJ * gene_pathway.shape[1]
    mask_raw = gene_pathway

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
                             beta=beta,
                             attn_dropout=attn_dropout,
                             ff_dropout=ff_dropout,
                             classifier_dropout=classifier_dropout).to(device)

    ckpt = torch.load(model_path,map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False

    ##############################################
    ############ Model Validation ################
    ##############################################
    model.eval()
    name_val = []
    predict_val = np.zeros([len(sample_name), label_dim])
    with torch.no_grad():
        for index, (data, name) in enumerate(val_loader):
            index += 1
            if index % 100 == 1:
                print(index)
            data = data.to(device)
            name_val.append(name[0])
            pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
            logits = model(pathway_network_batch, data.permute(0, 2, 1), output_attentions=False)
            if (index) * BATCH_SIZE <= len(sample_name):
                predict_val[(index - 1) * BATCH_SIZE:(index) * BATCH_SIZE, :] = logits.data.cpu().numpy()
            else:
                predict_val[(index - 1) * BATCH_SIZE:, :] = logits.data.cpu().numpy()

    predict_columns=[]
    predict_data_val = pd.DataFrame(columns={'sample_id', 'predict_score_0'})
    predict_data_val['sample_id'] = name_val
    for i in range(predict_val.shape[1]):
        predict_data_val['predict_score_'+str(i)] = np.array(predict_val)[:, i]
        predict_columns.append('predict_score_'+str(i))
    predict_data_val=predict_data_val[['sample_id']+predict_columns]
    predict_data_val.to_csv(save_path + '/predict_score.txt', sep='\t', index=False)

    if evaluate:
        if label_path is None:
            print('Missing label data')
        else:
            label = pd.read_csv(label_path, sep='\t')
            predict_data_val=pd.merge(predict_data_val,label,on='sample_id',how='left')
            if label_dim == 2:
                f = open(save_path + '/result_predict_evaluate.txt', 'w')
                ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(np.array(predict_data_val['y']), np.array(predict_data_val['predict_score_1']))
                # output
                f.write('ACC_val:' + str(ACC_val) + '\n')
                f.write('auc_val:' + str(AUC_val) + '\n')
                f.write('f1_weighted_val:' + str(f1_weighted_val) + '\n')
                f.write('f1_macro_val:' + str(f1_macro_val) + '\n')
                print('ACC_val:', ACC_val)
                print('auc_val:', AUC_val)
                print('f1_weighted_val:', f1_weighted_val)
                print('f1_macro_val:', f1_macro_val)
                f.close()
            else:
                y_val=np.array(predict_data_val['y'])
                predict_val=np.array(predict_data_val[predict_columns])
                f = open(save_path + '/result_predict_evaluate.txt', 'w')
                acc_val, auc_weighted_ovr_val, auc_weighted_ovo_val, auc_macro_ovr_val, auc_macro_ovo_val, f1_weighted_val, f1_macro_val = get_roc_multi(y_val, predict_val)
                y_val_new = np.array(y_val).copy()
                y_val_new[y_val_new >= 1] = 1
                ACC_val_2, AUC_val_2, f1_weighted_val_2, f1_macro_val_2 = get_roc(np.array(y_val_new),1 - np.array(predict_val)[:, 0])

                # output
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
                f.close()



def main(args):
    main_validation(args.modal_all_path,args.modal_select_path,args.gene_all, args.gene_select, args.pathway_gene_w, args.pathway_crosstalk_network,
                    args.data_path, args.label_path, args.sample_name_path,args.model_path, args.save_path, args.label_dim, args.evaluate,
                    args.depth, args.heads, args.dim_head, args.beta, args.attn_dropout, args.ff_dropout, args.classifier_dropout)


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
    parser.add_argument('--sample_name_path', type=str, required=True,
                        help='sample_name_path', dest='sample_name_path')
    parser.add_argument('--model_path', type=str, required=True,
                        help='model_path', dest='model_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--label_dim', type=int, required=True,
                        help='label_dim', dest='label_dim')
    parser.add_argument('--evaluate', type=bool, required=True,
                        help='evaluate', dest='evaluate')
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
    args = parser.parse_args()
    main(args)


