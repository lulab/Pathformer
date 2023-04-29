import pandas as pd

data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_masked_seg.csv',sep=',')
data=data[['Sample','Chromosome','Start', 'End','Num_Probes','Segment_Mean']]
data['type']=data['Sample'].map(lambda x:x.split('-')[-1])
data_=data.loc[data['type']=='01',:]
data_=data_[['Sample','Chromosome','Start', 'End','Num_Probes','Segment_Mean']]
data_.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_masked_seg_filter.txt',sep='\t',index=False)
