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
import h5py

from Pathformer import pathformer_model
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_attn(attn_out):
    attn_out_new = []
    for dots in attn_out:
        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn_out_new.append(attn)
    attn_weights_new = torch.cat(attn_out_new, dim=1)
    attn_weights_mean = torch.mean(attn_weights_new, dim=1)
    return attn_weights_mean



def get_attention_map(modal_all_path,modal_select_path,gene_all,gene_select,pathway_gene_w,pathway_crosstalk_network,
                      model_path,data_path,label_path,dataset,save_path,depth,heads,dim_head,beta,attn_dropout,ff_dropout,classifier_dropout):
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

    if (modal_all_path == 'None') | (modal_select_path == 'None'):
        modal_select_index = list(range(data.shape[2]))
    else:
        modal_all_data = pd.read_csv(modal_all_path, header=None)
        modal_all_data.columns = ['modal_type']
        modal_all_data['index'] = range(len(modal_all_data))
        modal_all_data = modal_all_data.set_index('modal_type')
        modal_select_data = pd.read_csv(modal_select_path, header=None)
        modal_select_index = list(modal_all_data.loc[list(modal_select_data[0]), 'index'])

    label = pd.read_csv(label_path, sep='\t')
    train_sample = list(label.loc[label['dataset_' + str(dataset) + '_test'] == 'train', :].index)
    test_sample = list(label.loc[label['dataset_' + str(dataset) + '_test'] == 'test', :].index)
    validation_sample = list(label.loc[label['dataset_' + str(dataset)] == 'validation', :].index)
    train_label = label.loc[train_sample, ['y']].values
    train_label = train_label.astype(int)
    test_label = label.loc[test_sample, ['y']].values
    test_label = test_label.astype(int)
    validation_label = label.loc[validation_sample, ['y']].values
    validation_label = validation_label.astype(int)

    data = np.load(file=data_path)
    data = data[train_sample + test_sample + validation_sample, :, :][:, gene_select_index, :][:, :, modal_select_index]
    label_all = np.concatenate([train_label, test_label, validation_label])
    data_dataset = SCDataset(data, label_all)
    data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)

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
    if N_PRJ == 1:
        embeding = True
        embeding_num = 32
        row_dim = embeding_num
        classifier_input = embeding_num * gene_pathway.shape[1]
    else:
        embeding = False
        row_dim = N_PRJ
        classifier_input = N_PRJ * gene_pathway.shape[1]
    mask_raw = gene_pathway
    label_dim = len(set(label['y']))

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

    ckpt = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False


    #######################
    ### attention out #####
    #######################

    gene_pathway=h5py.File(save_path+'/gene_pathway.h5', 'w')
    gene_pathway['weight']=ckpt['model_state_dict']['pathway_model.weight'].data.cpu().numpy()
    gene_pathway.close()

    attn_row = h5py.File(save_path+'/attn_out_row_all.h5', 'a')
    attn_col = h5py.File(save_path+'/attn_out_col_all.h5', 'a')
    net_all = h5py.File(save_path+'/net_all.h5', 'a')
    y_val = []

    model.eval()
    with torch.no_grad():
        for batch_index, (data, labels) in enumerate(data_loader):
            print(batch_index)
            batch_index += 1
            data = data.to(device)
            y = labels.to(device)
            y = y[:, 0]
            pathway_network_batch = repeat(pathway_network, 'i j-> x i j', x=data.shape[0])
            dec_logits, _, attn_out_row_list, _, attn_out_col_list, net= model(pathway_network_batch,data.permute(0, 2, 1),output_attentions=True)
            attn_out_row = get_attn(attn_out_row_list)
            attn_out_col = get_attn(attn_out_col_list)

            attn_row[str(batch_index)] = attn_out_row.data.cpu().numpy()
            attn_col[str(batch_index)] = attn_out_col.data.cpu().numpy()
            net_all[str(batch_index)] = net[0, :, :].data.cpu().numpy()
            y_val.append(y.tolist())


    np.save(file=save_path+'/data_label.npy', arr=np.array(y_val))
    attn_row.close()
    attn_col.close()
    net_all.close()

def main(args):
    get_attention_map(args.modal_all_path, args.modal_select_path, args.gene_all, args.gene_select, args.pathway_gene_w,args.pathway_crosstalk_network,
                      args.model_path, args.data_path, args.label_path, args.dataset, args.save_path, args.depth, args.heads, args.dim_head,
                      args.beta, args.attn_dropout,args.ff_dropout, args.classifier_dropout)


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
    parser.add_argument('--model_path', type=str, required=True,
                        help='model_path', dest='model_path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
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


