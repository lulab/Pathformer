import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import MinMaxScaler,StandardScaler

path='/qhsky1/liuxiaofan_result/model_TCGA_new/BRCA_subtype/7.CCnet_new/10/final_save/'
label = pd.read_csv('/qhsky1/liuxiaofan/Data/TCGA_new/BRCA_subtype/sample_id/sample_cross_subtype_new_new.txt', sep='\t')
data_label=np.load(file=path+'/data_label.npy')
data_label=pd.DataFrame(data_label)
data_label.columns=['y']
label_list=list(set(data_label['y']))
attn_col=h5py.File(path+'/attn_out_col_2_all.h5', 'r')

omics_name=['RNA_all_TPM',
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
data=pd.DataFrame(columns={'omics'})
data['omics']=omics_name
for y in label_list:
    type=list(label.loc[label['y']==y,'subtype'])[0]
    label_index=list(data_label.loc[data_label['y']==y,:].index+1)
    attn_col_all=np.zeros([len(label_index),len(omics_name)])
    for s in range(len(label_index)):
        if s%100==0:
            print(s)
        attn_col_=attn_col[str(label_index[s])][:][0]
        attn_col_=attn_col_.mean(axis=0)
        attn_col_all[s,:]=attn_col_
    attn_col_important= attn_col_all.mean(axis=0)
    data[type]=attn_col_important

label_index=list(data_label.index+1)
attn_col_all=np.zeros([len(label_index),len(omics_name)])
for s in range(len(label_index)):
    if s%100==0:
        print(s)
    attn_col_=attn_col[str(label_index[s])][:][0]
    attn_col_=attn_col_.mean(axis=0)
    attn_col_all[s,:]=attn_col_
attn_col_important= attn_col_all.mean(axis=0)
data['all']=attn_col_important

RNA=['RNA_all_TPM']
DNA_methylation=['methylation_count',
'methylation_max',
'methylation_min',
'methylation_mean',
'promoter_methylation_count',
'promoter_methylation_max',
'promoter_methylation_min',
'promoter_methylation_mean']
CNV=['CNV_count',
'CNV_max',
'CNV_min',
'CNV_mean',
'CNV_gene_level']

data.to_csv(path+'important_omics.txt',sep='\t',index=False)

color_list=['#79acdb','#f9eabd','#f7d688','#fae742','#f3ab4c','#f58814','#aa5211','#883313','#500a0a'
    ,'#e5f1e0','#b1cba7','#7ea671','#4b813c','#0a5e01']


plt.figure(figsize=(7.5,5)) #?????????????????????
labels = list(data['omics']) #???????????????????????????/??????
sizes = list(data['all']) #???????????????????????????????????????
colors = color_list #?????????????????????
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0)
patches = plt.pie(sizes,
                  explode=explode,
                  labels=None,
                  labeldistance=1.2,  # ??????????????????????????????
                  colors=colors,
                  # autopct =None ,  #???????????????????????????'%3.2f%%'
                  shadow = False,  #???????????????
                  startangle =90,  #???????????????????????????
                  pctdistance = None) #?????????????????????????????????
#patches?????????????????????texts1????????????label????????????texts2?????????????????????
plt.axis('equal')
plt.legend()
# plt.show()
plt.savefig('/qhsky1/liuxiaofan_result/model_TCGA_new/plot/result/F4/BRCA_subtype_pie.pdf')
plt.close()