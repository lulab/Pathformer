import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import MinMaxScaler,StandardScaler


d=8
###数据读入###
pathway_mx = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathway_w_filter_rank.npy')
pathway_mx[np.isnan(pathway_mx)] = 0
pathway_data=pd.read_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathways_select_filter_rank.txt',sep='\t')
pathway_name=list(pathway_data['name'])
pathway_mx=pd.DataFrame(pathway_mx)
pathway_mx.columns=pathway_name
pathway_mx.index=pathway_name

label = pd.read_csv('/qhsky1/liuxiaofan/Data/TCGA_new/BRCA_subtype/sample_id/sample_cross_subtype_new_new.txt', sep='\t')
data=pd.read_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/'+str(d)+'/final_save/shap_important_pathway.txt',sep='\t')
pathway_100=list(data.loc[data['length']<=100,'name'])

data_label=np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/'+str(d)+'/final_save/data_label.npy')
data_label=pd.DataFrame(data_label)
data_label.columns=['y']
label_list=list(set(data_label['y']))
net_data=h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/'+str(d)+'/final_save/net_all.h5', 'r')

type='all'
net_all_new=np.load('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/'+str(d)+'/final_save/pathway_net_'+type+'_new.npy')
net_all_new=pd.DataFrame(net_all_new)
net_all_new.columns=pathway_name
net_all_new.index = pathway_name
net_all_new_select=net_all_new.loc[pathway_100,pathway_100]

###重要module####
cut_off = np.quantile(np.array(net_all_new_select), 0.997)
data_all=pd.DataFrame(columns={'pathway','score','number','pathway_corr'})
data_all['pathway']=pathway_100
for i in range(len(net_all_new_select)):
    net_all_=net_all_new_select.loc[:,net_all_new_select.iloc[i,:]>cut_off]
    net_all_select = net_all_.loc[(net_all_ > cut_off).sum(axis=1)>0, :]
    pathway_select=list(set(net_all_select.index)|set(net_all_select.columns))
    print(len(pathway_select))
    data_all.loc[i,'score']=data.loc[data.name.isin(pathway_select),'all_new'].mean()
    data_all.loc[i,'number']=len(pathway_select)
    data_all.loc[i, 'pathway_corr'] = ','.join(pathway_select)
    gene_select = ','.join(list(data.loc[data.name.isin(pathway_select),'gene']))
    data_all.loc[i, 'gene_number'] =len(set(gene_select.split(',')))

data_all=data_all.loc[(data_all['number']>1),:]
data_all=data_all.sort_values('score',ascending=False)
pathway_select_center=list(data_all['pathway'])[0]
pathway_select_new=list(data_all['pathway_corr'])[0].split(',')

data_weigth=pd.DataFrame(columns=['node1','node2','net_new','net_old'])
i=0
for p1 in pathway_select_new:
    for p2 in pathway_select_new:
        data_weigth.loc[i,'node1']=p1
        data_weigth.loc[i, 'node2'] = p2
        data_weigth.loc[i, 'net_new'] = abs(net_all_new.loc[p1,p2])
        data_weigth.loc[i, 'net_old'] = abs(pathway_mx.loc[p1, p2])
        i = i + 1

data_weigth['net_new_0']=data_weigth['net_new'].map(lambda x:1 if x>cut_off else 0)
data_weigth.to_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/plot/result/F4/BRCA_subtype/shap_important_net_all_0.997.txt',sep='\t',index=False)
pd.DataFrame(pathway_select_new).to_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/plot/result/F4/BRCA_subtype/shap_important_net_all_0.997_pathway.txt',sep='\t',index=False,header=False)

data_weigth_=data_weigth.loc[data_weigth['net_new_0']>0,:]
data_weigth_.index=range(len(data_weigth_))

data_weigth_new=pd.DataFrame(columns={'Source','Target','weight','pn'})
data_weigth_new['Source']=data_weigth_['node1']
data_weigth_new['Target']=data_weigth_['node2']
data_weigth_new['weight']=data_weigth_['net_new']

for i in range(len(data_weigth_)):
    if  (data_weigth_.loc[i,'net_new_0']==1)&(data_weigth_.loc[i,'net_old']==0):
        data_weigth_new.loc[i, 'pn'] = 1
    else:
        data_weigth_new.loc[i, 'pn'] = 0
data_weigth_new=data_weigth_new[['Source','Target','weight','pn']]
data_weigth_new.to_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/plot/result/F4/BRCA_subtype/shap_important_net_all_0.997_new.csv',sep=',',index=False)

