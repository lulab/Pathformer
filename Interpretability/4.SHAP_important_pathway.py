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

label = pd.read_csv('/qhsky1/liuxiaofan/Data/TCGA_new/BRCA_subtype/sample_id/sample_cross_subtype_new_new.txt', sep='\t')
pathway_data = pd.read_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/pathways_select_filter_rank.txt', sep='\t')
pathway_name = list(pathway_data['name'])
gene_data=pd.read_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/pathway/gene_select_filter_name.txt',sep='\t')
gene_name=list(gene_data['gene_name'])
# gene_name=list(gene_data['gene_id'])

data_label = np.load(file='/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/data_label_SHAP_pathway.npy')
data_label = pd.DataFrame(data_label)
data_label.columns = ['y']
label_list = list(set(data_label['y']))
shap_1 = h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_pathway_1.h5', 'r')
shap_2 = h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_pathway_2.h5', 'r')
shap_3 = h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_pathway_3.h5', 'r')
shap_4 = h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_pathway_4.h5', 'r')
shap_5 = h5py.File('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_pathway_5.h5', 'r')

label_index_1 = list(data_label.loc[data_label['y'] == 0, :].index + 1)
shap_1_all = np.zeros([len(label_index_1), 14, len(pathway_name)])
for s in range(len(label_index_1)):
    if s % 100 == 0:
        print(s)
    shap_1_all[s, :, :] = shap_1[str(s + 1)][:][0]

label_index_2 = list(data_label.loc[data_label['y'] == 1, :].index + 1)
shap_2_all = np.zeros([len(label_index_2), 14, len(pathway_name)])
for s in range(len(label_index_2)):
    if s % 100 == 0:
        print(s)
    shap_2_all[s, :, :] = shap_2[str(s + 1)][:][0]

label_index_3 = list(data_label.loc[data_label['y'] == 2, :].index + 1)
shap_3_all = np.zeros([len(label_index_3), 14, len(pathway_name)])
for s in range(len(label_index_3)):
    if s % 100 == 0:
        print(s)
    shap_3_all[s, :, :] = shap_3[str(s + 1)][:][0]

label_index_4 = list(data_label.loc[data_label['y'] == 3, :].index + 1)
shap_4_all = np.zeros([len(label_index_4), 14, len(pathway_name)])
for s in range(len(label_index_4)):
    if s % 100 == 0:
        print(s)
    shap_4_all[s, :, :] = shap_4[str(s + 1)][:][0]

label_index_5 = list(data_label.loc[data_label['y'] == 4, :].index + 1)
shap_5_all = np.zeros([len(label_index_5), 14, len(pathway_name)])
for s in range(len(label_index_5)):
    if s % 100 == 0:
        print(s)
    shap_5_all[s, :, :] = shap_5[str(s + 1)][:][0]

label_index_1 = list(data_label.loc[data_label['y'] >=0, :].index + 1)
shap_1_all_new = np.zeros([len(label_index_1), 14, len(pathway_name)])
for s in range(len(label_index_1)):
    if s % 100 == 0:
        print(s)
    shap_1_all_new[s, :, :] = shap_1[str(s + 1)][:][0]

label_index_2 = list(data_label.loc[data_label['y'] >=0, :].index + 1)
shap_2_all_new = np.zeros([len(label_index_2), 14, len(pathway_name)])
for s in range(len(label_index_2)):
    if s % 100 == 0:
        print(s)
    shap_2_all_new[s, :, :] = shap_2[str(s + 1)][:][0]

label_index_3 = list(data_label.loc[data_label['y'] >=0, :].index + 1)
shap_3_all_new = np.zeros([len(label_index_3), 14, len(pathway_name)])
for s in range(len(label_index_3)):
    if s % 100 == 0:
        print(s)
    shap_3_all_new[s, :, :] = shap_3[str(s + 1)][:][0]

label_index_4 = list(data_label.loc[data_label['y'] >=0, :].index + 1)
shap_4_all_new = np.zeros([len(label_index_4), 14, len(pathway_name)])
for s in range(len(label_index_4)):
    if s % 100 == 0:
        print(s)
    shap_4_all_new[s, :, :] = shap_4[str(s + 1)][:][0]

label_index_5 = list(data_label.loc[data_label['y'] >=0, :].index + 1)
shap_5_all_new = np.zeros([len(label_index_5), 14, len(pathway_name)])
for s in range(len(label_index_5)):
    if s % 100 == 0:
        print(s)
    shap_5_all_new[s, :, :] = shap_5[str(s + 1)][:][0]

shap_1.close()
shap_2.close()
shap_3.close()
shap_4.close()
shap_5.close()


shap_1_mean = np.mean(np.abs(shap_1_all), axis=0)
shap_2_mean = np.mean(np.abs(shap_2_all), axis=0)
shap_3_mean = np.mean(np.abs(shap_3_all), axis=0)
shap_4_mean = np.mean(np.abs(shap_4_all), axis=0)
shap_5_mean = np.mean(np.abs(shap_5_all), axis=0)
shap_1_mean_new = np.mean(np.abs(shap_1_all_new), axis=0)
shap_2_mean_new = np.mean(np.abs(shap_2_all_new), axis=0)
shap_3_mean_new = np.mean(np.abs(shap_3_all_new), axis=0)
shap_4_mean_new = np.mean(np.abs(shap_4_all_new), axis=0)
shap_5_mean_new = np.mean(np.abs(shap_5_all_new), axis=0)

data = pd.DataFrame(columns=['all'])
data['shap_1'] = shap_1_mean.sum(axis=0)
data['shap_2'] = shap_2_mean.sum(axis=0)
data['shap_3'] = shap_3_mean.sum(axis=0)
data['shap_4'] = shap_4_mean.sum(axis=0)
data['shap_5'] = shap_5_mean.sum(axis=0)
data['shap_1_new'] = shap_1_mean_new.sum(axis=0)
data['shap_2_new'] = shap_2_mean_new.sum(axis=0)
data['shap_3_new'] = shap_3_mean_new.sum(axis=0)
data['shap_4_new'] = shap_4_mean_new.sum(axis=0)
data['shap_5_new'] = shap_5_mean_new.sum(axis=0)
data['name'] = pathway_name

data['all']=data['shap_1']+data['shap_2']+data['shap_3']+data['shap_4']+data['shap_5']
data['all_new']=data['shap_1_new']+data['shap_2_new']+data['shap_3_new']+data['shap_4_new']+data['shap_5_new']

shap_mean=shap_1_mean_new+shap_2_mean_new+shap_3_mean_new+shap_4_mean_new+shap_5_mean_new

data=pd.merge(data,pathway_data[['name','gene','length']],on='name',how='left')
data.to_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_important_pathway.txt',sep='\t',index=False)

data_selet=data.loc[data['length']<100,:].sort_values('all_new',ascending=False)
important_pathway=list(data_selet['name'])[:15]
pd.DataFrame(important_pathway).to_csv('/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/8/final_save/shap_important_pathway_top15.txt',sep='\t',index=False,header=False)


shap_mean=pd.DataFrame(shap_mean.T)
shap_mean.columns=['RNA_all_TPM',
'methylation_count',
'methylation_max',
'methylation_min',
'methylation_mean',
'promoter_methylation_count',
'promoter_methylation_max',
'promoter_methylation_min',
'promoter_methylation_mean',
'CNV_count',
'CNV_max',
'CNV_min',
'CNV_mean',
'CNV_gene_level']
shap_mean.index=pathway_name

shap_mean['RNA']=shap_mean['RNA_all_TPM']
shap_mean['DNA']=0
for i in ['methylation_count','methylation_max','methylation_min','methylation_mean','promoter_methylation_count',
          'promoter_methylation_max','promoter_methylation_min','promoter_methylation_mean']:
    shap_mean['DNA'] = shap_mean['DNA']+shap_mean[i]
shap_mean['DNA']=shap_mean['DNA']

shap_mean['CNV']=0
for i in ['CNV_count','CNV_max','CNV_min','CNV_mean','CNV_gene_level']:
    shap_mean['CNV'] = shap_mean['CNV']+shap_mean[i]
shap_mean['CNV']=shap_mean['CNV']

data_import_select=shap_mean.loc[important_pathway,['RNA','DNA','CNV']]
data_import_select.loc[data_import_select['RNA']>0.005,'RNA']=0.005

preprocess = MinMaxScaler()
data_import_select_1 = pd.DataFrame(preprocess.fit_transform(data_import_select))
data_import_select_1.index=important_pathway
data_import_select_1.columns=data_import_select.columns



fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(3, 5), sharey=True)
fig.subplots_adjust(wspace=0.03)
ax1 = sns.heatmap(data_import_select_1[['RNA']],cmap=sns.color_palette("light:b", as_cmap=True),linewidth=0.3,cbar=False,yticklabels=True,ax=ax[0])
ax2 = sns.heatmap(data_import_select_1[['DNA']],cmap=sns.color_palette("light:r", as_cmap=True),linewidth=0.3,cbar=False,yticklabels=True, ax=ax[1])
ax3 = sns.heatmap(data_import_select_1[['CNV']],cmap=sns.light_palette("seagreen", as_cmap=True),linewidth=0.3,cbar=False,yticklabels=True, ax=ax[2])
plt.tight_layout()
plt.savefig('/qhsky1/liuxiaofan_result/model_TCGA_new/plot/result/F4/BRCA_subtype/all.pdf')
# plt.show()
plt.close()

print(list(data_import_select_1.index))