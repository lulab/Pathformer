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
import shap

from CustomizedLinear import CustomizedLinear
from Pathformer import Evoformer,classifier
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_shap_pathway_sample(modal_all_path, modal_select_path, gene_all, gene_select, pathway_gene_w,pathway_crosstalk_network,
                      model_path, data_path, label_path, dataset, save_path, depth, heads, dim_head, beta, attn_dropout,
                      ff_dropout, classifier_dropout):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_seed(2022)

    PATHWAY_CROSSTALK_NETWORK=pathway_crosstalk_network
    class pathformer_model(nn.Module):
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
                     embeding,
                     embeding_num,
                     beta,
                     attn_dropout,
                     ff_dropout,
                     classifier_dropout):
            super(pathformer_model, self).__init__()
            self.pathway_model = CustomizedLinear(mask_raw, bias=None)
            self.Evoformer_model = Evoformer(depth=depth, dim_network=1, row_dim=row_dim,
                                         col_dim=col_dim, heads=heads,dim_head=dim_head,embeding=embeding,embeding_num=embeding_num,
                                         beta=beta,attn_dropout=attn_dropout,ff_dropout=ff_dropout)
            self.classifier_model = classifier(classifier_input, classifier_dim, label_dim, classifier_dropout)

        def forward(self,m_row):
            pathway_network = np.load(file=PATHWAY_CROSSTALK_NETWORK)
            pathway_network[np.isnan(pathway_network)] = 0
            pathway_network = torch.Tensor(pathway_network).to(device)
            x = repeat(pathway_network, 'i j-> x i j', x=m_row.shape[0])
            m = self.pathway_model(m_row)
            x, m = self.Evoformer_model(x, m, output_attentions=False)
            dec_logits = self.classifier_model(m)

            return dec_logits


    #############################
    #########Data load###########
    #############################
    ###Sample data load
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
    validation_sample = list(label.loc[label['dataset_' + str(dataset)] == 'validation', :].index)
    test_sample = list(label.loc[label['dataset_' + str(dataset) + '_test'] == 'test', :].index)
    train_label = label.loc[train_sample, ['y']].values
    train_label = train_label.astype(int)
    validation_label = label.loc[validation_sample, ['y']].values
    validation_label = validation_label.astype(int)
    test_label = label.loc[test_sample, ['y']].values
    test_label = test_label.astype(int)


    data = np.load(file=data_path)
    data = data[train_sample+ validation_sample + test_sample , :, :][:, gene_select_index, :][:, :, modal_select_index]
    label_all = np.concatenate([train_label, validation_label, test_label])
    data_dataset = SCDataset(data, label_all)
    train_loader = DataLoader(data_dataset, batch_size=40, num_workers=0, pin_memory=True, shuffle=True)
    test_loader = DataLoader(data_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)

    ###Pathway crosstalk netwark load
    gene_pathway = np.load(file=pathway_gene_w)
    gene_pathway = torch.LongTensor(gene_pathway)


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
                             embeding_num=embeding_num,
                             beta=beta,
                             attn_dropout=attn_dropout,
                             ff_dropout=ff_dropout,
                             classifier_dropout=classifier_dropout).to(device)

    ckpt = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False

    #######################
    ####### SHAP out ######
    #######################
    data_train = next(iter(train_loader))
    (input_train, targets_train) = data_train
    background = input_train.permute(0, 2, 1).float().to(device)
    explainer = shap.GradientExplainer((model,model.classifier_model),background)

    N_SAMPLES=1
    shap_all = h5py.File(save_path+'/shap_pathway_all.h5', 'a')
    for i in range(label_dim):
        shap_all.create_group('group'+'_'+str(i))
    y_all = []

    model.eval()
    for batch_index, (data, labels) in enumerate(test_loader):
        print(batch_index)
        batch_index += 1
        data = data.to(device)
        y = labels.to(device)
        y = y[:, 0]
        shap_= explainer.shap_values(data.permute(0, 2, 1), nsamples=N_SAMPLES)
        for i in range(len(shap_)):
            shap_all['group'+'_'+str(i)][str(batch_index)]=shap_[i]
        y_all.append(y.tolist())


    np.save(file=save_path + '/data_label_SHAP_pathway.npy', arr=np.array(y_all))
    shap_all.close()

def main(args):
    get_shap_pathway_sample(args.modal_all_path, args.modal_select_path, args.gene_all, args.gene_select, args.pathway_gene_w,args.pathway_crosstalk_network,
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




