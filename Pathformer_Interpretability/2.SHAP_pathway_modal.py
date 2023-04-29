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


def get_shap_pathway(modal_all_path,modal_select_path,pathway_name_path,save_path):
    pathway_data = pd.read_csv(pathway_name_path, sep='\t')
    pathway_name = list(pathway_data['name'])


    data_label = np.load(file=save_path+'/data_label_SHAP_pathway.npy')
    data_label = pd.DataFrame(data_label)
    data_label.columns = ['y']
    label_list = list(set(data_label['y']))

    shap_all = h5py.File(save_path+'/shap_pathway_all.h5', 'r')
    label_index = list(data_label.index + 1)
    data = pd.DataFrame(columns=['pathway'])
    data['pathway'] = pathway_name
    data['shap_pathway']=0
    shap_pathway_modal=np.zeros([shap_all['group_0']['1'].shape[1], shap_all['group_0']['1'].shape[2]])

    for i in range(len(set(data_label['y']))):
        shap_new=np.zeros([len(label_index), shap_all['group_0']['1'].shape[1], shap_all['group_0']['1'].shape[2]])
        for s in range(len(label_index)):
            if s % 100 == 0:
                print(s)
            shap_new[s, :, :] = shap_all['group_'+str(i)][str(s + 1)][:][0]
        shap_mean = np.mean(np.abs(shap_new), axis=0)
        shap_pathway_modal=shap_pathway_modal+shap_mean
        data['shap_group_'+str(i)]=shap_mean.sum(axis=0)
        data['shap_pathway']=data['shap_pathway']+data['shap_group_'+str(i)]
    shap_all.close()

    data=pd.merge(data,pathway_data[['name','gene','length']].rename(columns={'name':'pathway'}),on='pathway',how='left')
    data.to_csv(save_path+'/shap_value_pathway.txt',sep='\t',index=False)

    data_selet=data.loc[data['length']<100,:].sort_values('shap_pathway',ascending=False)
    important_pathway=list(data_selet['pathway'])[:15]
    pd.DataFrame(important_pathway).to_csv(save_path+'/shap_important_pathway_top15.txt',sep='\t',index=False,header=False)


    modal_name_all=list(pd.read_csv(modal_all_path,header=None)[0])
    modal_name_select=list(pd.read_csv(modal_select_path,header=None)[0])

    shap_pathway_modal=pd.DataFrame(shap_pathway_modal.T)
    shap_pathway_modal.columns=modal_name_select
    shap_pathway_modal.index=pathway_name
    modal_selet=[]

    if len(set(RNA)&set(modal_name_select))>0:
        shap_pathway_modal['RNA'] = shap_pathway_modal[RNA[0]]
        modal_selet.append('RNA')
    if len(set(DNA_methylation)&set(modal_name_select))>0:
        shap_pathway_modal['DNA_methylation']=0
        for i in DNA_methylation:
            shap_pathway_modal['DNA_methylation'] = shap_pathway_modal['DNA_methylation']+shap_pathway_modal[i]
        shap_pathway_modal['DNA_methylation']=shap_pathway_modal['DNA_methylation']
        modal_selet.append('DNA_methylation')
    if len(set(DNA_CNV)&set(modal_name_select))>0:
        shap_pathway_modal['DNA_CNV']=0
        for i in DNA_CNV:
            shap_pathway_modal['DNA_CNV'] = shap_pathway_modal['DNA_CNV']+shap_pathway_modal[i]
        shap_pathway_modal['DNA_CNV']=shap_pathway_modal['DNA_CNV']
        modal_selet.append('DNA_CNV')
    if len(set(RNA_expression)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_expression'] = shap_pathway_modal[RNA_expression[0]]
        modal_selet.append('RNA_expression')
    if len(set(RNA_chimeric)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_chimeric'] = shap_pathway_modal[RNA_chimeric[0]]
        modal_selet.append('RNA_chimeric')
    if len(set(RNA_promoter)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_promoter']=0
        for i in RNA_promoter:
            shap_pathway_modal['RNA_promoter'] = shap_pathway_modal['RNA_promoter']+shap_pathway_modal[i]
        shap_pathway_modal['RNA_promoter']=shap_pathway_modal['RNA_promoter']
        modal_selet.append('RNA_promoter')
    if len(set(RNA_splicing)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_splicing']=0
        for i in RNA_splicing:
            shap_pathway_modal['RNA_splicing'] = shap_pathway_modal['RNA_splicing']+shap_pathway_modal[i]
        shap_pathway_modal['RNA_splicing']=shap_pathway_modal['RNA_splicing']
        modal_selet.append('RNA_splicing')
    if len(set(RNA_ASE)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_ASE']=0
        for i in RNA_ASE:
            shap_pathway_modal['RNA_ASE'] = shap_pathway_modal['RNA_ASE']+shap_pathway_modal[i]
        shap_pathway_modal['RNA_ASE']=shap_pathway_modal['RNA_ASE']
        modal_selet.append('RNA_ASE')
    if len(set(RNA_editing)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_editing']=0
        for i in RNA_editing:
            shap_pathway_modal['RNA_editing'] = shap_pathway_modal['RNA_editing']+shap_pathway_modal[i]
        shap_pathway_modal['RNA_editing']=shap_pathway_modal['RNA_editing']
        modal_selet.append('RNA_editing')
    if len(set(RNA_SNP)&set(modal_name_select))>0:
        shap_pathway_modal['RNA_SNP']=0
        for i in RNA_SNP:
            shap_pathway_modal['RNA_SNP'] = shap_pathway_modal['RNA_SNP']+shap_pathway_modal[i]
        shap_pathway_modal['RNA_SNP']=shap_pathway_modal['RNA_SNP']
        modal_selet.append('RNA_SNP')


    data_import_select=shap_pathway_modal.loc[important_pathway,modal_selet]
    preprocess = MinMaxScaler()
    data_import_select_new = pd.DataFrame(preprocess.fit_transform(data_import_select))
    data_import_select_new.index=important_pathway
    data_import_select_new.columns=data_import_select.columns

    shap_pathway_modal.to_csv(save_path+'/shap_pathway_modal.txt',sep='\t',index=True)
    data_import_select_new.to_csv(save_path+'/shap_important_pathway_modal.txt',sep='\t',index=True)

def main(args):
    get_shap_pathway(args.modal_all_path,args.modal_select_path,args.pathway_name_path,args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--modal_all_path', type=str, default='None',
                        help='modal_all_path', dest='modal_all_path')
    parser.add_argument('--modal_select_path', type=str, default='None',
                        help='modal_select_path', dest='modal_select_path')
    parser.add_argument('--pathway_name_path', type=str, required=True,
                        help='pathway_name_path', dest='pathway_name_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)
