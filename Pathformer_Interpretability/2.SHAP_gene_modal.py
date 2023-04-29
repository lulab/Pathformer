import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import sys

from modal_type_list import *
import argparse


def get_shap_gene(modal_all_path,modal_select_path,pathway_name_path,gene_name_path,save_path):

    pathway_data = pd.read_csv(pathway_name_path, sep='\t')
    pathway_name = list(pathway_data['name'])
    gene_data=pd.read_csv(gene_name_path,sep='\t')
    gene_name=list(gene_data['gene_name'])

    data_label = np.load(file=save_path+'/data_label_SHAP.npy')
    data_label = pd.DataFrame(data_label)
    data_label.columns = ['y']
    label_list = list(set(data_label['y']))
    shap_gene_all = h5py.File(save_path+'/shap_gene_all.h5', 'r')

    label_index = list(data_label.index + 1)
    data = pd.DataFrame(columns=['gene'])
    data['gene'] = gene_name
    data['shap_gene']=0
    shap_gene_modal=np.zeros([shap_gene_all['group_0']['1'].shape[1], shap_gene_all['group_0']['1'].shape[2]])

    for i in range(len(set(data_label['y']))):
        shap_new=np.zeros([len(label_index), shap_gene_all['group_0']['1'].shape[1], shap_gene_all['group_0']['1'].shape[2]])
        for s in range(len(label_index)):
            if s % 100 == 0:
                print(s)
            shap_new[s, :, :] = shap_gene_all['group_'+str(i)][str(s + 1)][:][0]
        shap_mean = np.mean(np.abs(shap_new), axis=0)
        shap_gene_modal=shap_gene_modal+shap_mean
        data['shap_gene_group_'+str(i)]=shap_mean.sum(axis=0)
        data['shap_gene']=data['shap_gene']+data['shap_gene_group_'+str(i)]
    shap_gene_all.close()

    modal_name_all=list(pd.read_csv(modal_all_path,header=None)[0])
    modal_name_select=list(pd.read_csv(modal_select_path,header=None)[0])

    shap_gene_modal = pd.DataFrame(shap_gene_modal.T)
    shap_gene_modal.columns=modal_name_select
    shap_gene_modal.index=gene_name

    modal_selet=[]

    if len(set(RNA)&set(modal_name_select))>0:
        shap_gene_modal['RNA'] = shap_gene_modal[RNA[0]]
        shap_gene_modal['RNA_rank'] = shap_gene_modal.RNA.rank(method='min', ascending=False)
        modal_selet.append('RNA')
    if len(set(DNA_methylation)&set(modal_name_select))>0:
        shap_gene_modal['DNA_methylation']=0
        for i in DNA_methylation:
            shap_gene_modal['DNA_methylation'] = shap_gene_modal['DNA_methylation']+shap_gene_modal[i]
        shap_gene_modal['DNA_methylation']=shap_gene_modal['DNA_methylation']
        shap_gene_modal['DNA_methylation_rank'] = shap_gene_modal.DNA_methylation.rank(method='min', ascending=False)
        modal_selet.append('DNA_methylation')
    if len(set(DNA_CNV)&set(modal_name_select))>0:
        shap_gene_modal['DNA_CNV']=0
        for i in DNA_CNV:
            shap_gene_modal['DNA_CNV'] = shap_gene_modal['DNA_CNV']+shap_gene_modal[i]
        shap_gene_modal['DNA_CNV']=shap_gene_modal['DNA_CNV']
        shap_gene_modal['DNA_CNV_rank'] = shap_gene_modal.DNA_CNV.rank(method='min', ascending=False)
        modal_selet.append('DNA_CNV')
    if len(set(RNA_expression)&set(modal_name_select))>0:
        shap_gene_modal['RNA_expression'] = shap_gene_modal[RNA_expression[0]]
        shap_gene_modal['RNA_expression_rank'] = shap_gene_modal.RNA_expression.rank(method='min', ascending=False)
        modal_selet.append('RNA_expression')
    if len(set(RNA_chimeric)&set(modal_name_select))>0:
        shap_gene_modal['RNA_chimeric'] = shap_gene_modal[RNA_chimeric[0]]
        shap_gene_modal['RNA_chimeric_rank'] = shap_gene_modal.RNA_chimeric.rank(method='min', ascending=False)
        modal_selet.append('RNA_chimeric')
    if len(set(RNA_promoter)&set(modal_name_select))>0:
        shap_gene_modal['RNA_promoter']=0
        for i in RNA_promoter:
            shap_gene_modal['RNA_promoter'] = shap_gene_modal['RNA_promoter']+shap_gene_modal[i]
        shap_gene_modal['RNA_promoter']=shap_gene_modal['RNA_promoter']
        shap_gene_modal['RNA_promoter_rank'] = shap_gene_modal.RNA_promoter.rank(method='min', ascending=False)
        modal_selet.append('RNA_promoter')
    if len(set(RNA_splicing)&set(modal_name_select))>0:
        shap_gene_modal['RNA_splicing']=0
        for i in RNA_splicing:
            shap_gene_modal['RNA_splicing'] = shap_gene_modal['RNA_splicing']+shap_gene_modal[i]
        shap_gene_modal['RNA_splicing']=shap_gene_modal['RNA_splicing']
        shap_gene_modal['RNA_splicing_rank'] = shap_gene_modal.RNA_splicing.rank(method='min', ascending=False)
        modal_selet.append('RNA_splicing')
    if len(set(RNA_ASE)&set(modal_name_select))>0:
        shap_gene_modal['RNA_ASE']=0
        for i in RNA_ASE:
            shap_gene_modal['RNA_ASE'] = shap_gene_modal['RNA_ASE']+shap_gene_modal[i]
        shap_gene_modal['RNA_ASE']=shap_gene_modal['RNA_ASE']
        shap_gene_modal['RNA_ASE_rank'] = shap_gene_modal.RNA_ASE.rank(method='min', ascending=False)
        modal_selet.append('RNA_ASE')
    if len(set(RNA_editing)&set(modal_name_select))>0:
        shap_gene_modal['RNA_editing']=0
        for i in RNA_editing:
            shap_gene_modal['RNA_editing'] = shap_gene_modal['RNA_editing']+shap_gene_modal[i]
        shap_gene_modal['RNA_editing']=shap_gene_modal['RNA_editing']
        shap_gene_modal['RNA_editing_rank'] = shap_gene_modal.RNA_editing.rank(method='min', ascending=False)
        modal_selet.append('RNA_editing')
    if len(set(RNA_SNP)&set(modal_name_select))>0:
        shap_gene_modal['RNA_SNP']=0
        for i in RNA_SNP:
            shap_gene_modal['RNA_SNP'] = shap_gene_modal['RNA_SNP']+shap_gene_modal[i]
        shap_gene_modal['RNA_SNP']=shap_gene_modal['RNA_SNP']
        shap_gene_modal['RNA_SNP_rank'] = shap_gene_modal.RNA_SNP.rank(method='min', ascending=False)
        modal_selet.append('RNA_SNP')


    important_pathway=list(pd.read_csv(save_path+'/shap_important_pathway_top15.txt',sep='\t',header=None)[0])
    shap_important_gene_modal=pd.DataFrame(columns=['pathway','gene','shap_gene','Pillar_modality']+modal_selet+[i+'_rank' for i in modal_selet])

    for p in important_pathway:
        # print(p)
        shap_important_gene_modal_=pd.DataFrame(columns=['pathway','gene','shap_gene','Pillar_modality']+modal_selet+[i+'_rank' for i in modal_selet])
        gene_ = list(pathway_data.loc[pathway_data['name'] == p, 'gene'])[0].split(',')
        gene_=list(set(gene_)&set(shap_gene_modal.index))
        data_ = data.loc[data.gene.isin(gene_)].sort_values('shap_gene', ascending=False)
        data_=data_.set_index('gene')
        gene_top5= list(data_.index)[:5]
        shap_important_gene_modal_['gene'] = gene_top5
        shap_important_gene_modal_['shap_gene']=list(data_.loc[gene_top5, 'shap_gene'])
        for i in modal_selet:
            shap_important_gene_modal_[i] = list(shap_gene_modal.loc[gene_top5, i])
        for r in [i+'_rank' for i in modal_selet]:
            shap_important_gene_modal_[r] = list(shap_gene_modal.loc[gene_top5, r])
        shap_gene_modal_ = shap_gene_modal.loc[gene_top5, [i+'_rank' for i in modal_selet]]
        for n in range(5):
            shap_gene_modal_1=shap_gene_modal_.loc[gene_top5[n],:]
            shap_important_gene_modal_.loc[n,'Pillar_modality']=','.join(list(shap_gene_modal_1[shap_gene_modal_1==shap_gene_modal_1.max()].index))
        shap_important_gene_modal_['pathway']=p
        shap_important_gene_modal=pd.concat([shap_important_gene_modal,shap_important_gene_modal_])

    shap_gene_modal[modal_selet].to_csv(save_path+'/shap_gene_modal.txt',sep='\t',index=True)
    shap_important_gene_modal.to_csv(save_path+'/shap_important_gene_modal.txt',sep='\t',index=False)

def main(args):
    get_shap_gene(args.modal_all_path,args.modal_select_path,args.pathway_name_path,args.gene_name_path,args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--modal_all_path', type=str, default='None',
                        help='modal_all_path', dest='modal_all_path')
    parser.add_argument('--modal_select_path', type=str, default='None',
                        help='modal_select_path', dest='modal_select_path')
    parser.add_argument('--pathway_name_path', type=str, required=True,
                        help='pathway_name_path', dest='pathway_name_path')
    parser.add_argument('--gene_name_path', type=str, required=True,
                        help='gene_name_path', dest='gene_name_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)

