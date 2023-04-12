import pandas as pd
import numpy as np
import re
import scipy.stats

data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/ACC.DNAmethy.csv',sep=',')
sample=data.columns[1:]
data.columns=['ID']+list(sample)

ID_gene=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/DNA_methylation_promoterid.txt',sep='\t')
data=data.loc[data.ID.isin(list(ID_gene['ID']))]
data=pd.merge(ID_gene[['ID','gene_id']],data,on='ID',how='left')

data=data.loc[pd.notnull(data['gene_id'])]
data[data==0]=np.nan
sample=list(data.columns[2:])

# @jit(nopython=False)
def get_entropy(x):
    entropy_list=[]
    for i in sample:
        arr=np.array(x[i])
        if len(set(arr))==1:
            entropy=0
        else:
            entropy=scipy.stats.entropy(arr[arr>0])
        entropy_list.append(entropy)
    data=pd.Series(entropy_list)
    data.index=sample
    return data

#max
data_max=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
data_max=data_max.fillna(0)
# data_max=data_max.reset_index()
data_max.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/promoter_methylation_max.txt',sep='\t',index=False)
del data_max
#min
data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
data_min=data_min.fillna(0)
# data_min=data_min.reset_index()
data_min.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/promoter_methylation_min.txt',sep='\t',index=False)
del data_min
#mean
data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
data_mean=data_mean.fillna(0)
data_mean=data_mean.reset_index()
data_mean.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/promoter_methylation_mean.txt',sep='\t',index=False)
del data_mean
#count
data=data.fillna(0)
data_=data[sample].copy()
data_[data_>0]=1
data_['gene_id']=data['gene_id']
data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
data_count=data_count.fillna(0)
data_count=data_count.reset_index()
data_count.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/promoter_methylation_count.txt',sep='\t',index=False)

