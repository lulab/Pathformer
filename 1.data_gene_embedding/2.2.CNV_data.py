import pandas as pd
import numpy as np
import re

data_CNV=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/BRCA.CNV_masked_seg.csv',sep=',')
data_CNV['ID']='chr'+data_CNV['Chromosome'].astype(str)+'_'+data_CNV['Start'].astype(str)+'_'+data_CNV['End'].astype(str)
sample=list(set(data_CNV['Sample']))

data=pd.DataFrame(columns=['ID'])
j=0
for i in sample:
    print(j)
    data_CNV_=data_CNV.loc[data_CNV['Sample']==i,['ID','Segment_Mean']].rename(columns={'Segment_Mean':i})
    data=pd.merge(data,data_CNV_,on='ID',how='outer')
    j=j+1
data.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/BRCA.CNV_masked_seg.all.csv',sep='\t',index=False)

ID_gene=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/CNV_id.txt',sep='\t')
ID_gene=ID_gene.drop_duplicates()
data=data.loc[data.ID.isin(list(ID_gene['ID']))]
data=pd.merge(ID_gene[['ID','gene_id']],data,on='ID',how='left')

del data_CNV
data=data.loc[pd.notnull(data['gene_id'])]
sample=list(data.columns[2:])

#count
data=data.fillna(0)
data_=data[sample].copy()
data_[data_>0]=1
data_[data_<0]=1
data_['gene_id']=data['gene_id']
data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
data_count=data_count.fillna(0)
data_count=data_count.reset_index()
data_count.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/CNV_count.txt',sep='\t',index=False)
del data_
del data_count


data_up=data[sample].copy()
data_up[data_up<=0]=np.nan
data_up['gene_id']=data['gene_id']
data_up_max=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
data_up_max.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_up_max.txt',sep='\t',index=False)
del data_up_max
data_up_min=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
data_up_min.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_up_min.txt',sep='\t',index=False)
del data_up_min
data_up_sum=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
data_up_sum=data_up_sum[sample]
data_up_sum.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_up_sum.txt',sep='\t',index=True)
del data_up_sum
data_up_mean=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
data_up_mean.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_up_mean.txt',sep='\t',index=True)
del data_up_mean
del data_up
data_down=data[sample].copy()
data_down[data_down>=0]=np.nan
data_down['gene_id']=data['gene_id']
data_down_max=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
data_down_max.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_down_max.txt',sep='\t',index=False)
del data_down_max
data_down_min=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
data_down_min.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_down_min.txt',sep='\t',index=False)
del data_down_min
data_down_sum=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
data_down_sum=data_down_sum[sample]
data_down_sum.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_down_sum.txt',sep='\t',index=True)
del data_down_sum
data_down_mean=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
data_down_mean.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/data_down_mean.txt',sep='\t',index=True)
del data_down_mean
del data_down
del data

