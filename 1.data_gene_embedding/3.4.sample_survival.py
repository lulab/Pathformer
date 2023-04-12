import pandas as pd

data_survival=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/survival_BRCA_survival.txt',sep='\t')

sample_mRNA_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_mRNA_data_2.txt',sep='\t')
sample_mRNA_data['sampleID']=sample_mRNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_DNA_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_DNA_data_2.txt',sep='\t')
sample_DNA_data['sampleID']=sample_DNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_CNV_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_CNV_data_2.txt',sep='\t')
sample_CNV_data['sampleID']=sample_CNV_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_CNV_gene_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_CNV_gene_data_2.txt',sep='\t')
sample_CNV_gene_data['sampleID']=sample_CNV_gene_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])

sample_all=set(sample_mRNA_data['sampleID'])&set(sample_DNA_data['sampleID'])&set(sample_CNV_data['sampleID'])&set(sample_CNV_gene_data['sampleID'])

for i in range(len(data_survival)):
    if (data_survival.loc[i,'OS']==1)&(data_survival.loc[i,'OS.time']<1825):
        data_survival.loc[i, 'label']=0
    elif (data_survival.loc[i,'OS']==1)&(data_survival.loc[i,'OS.time']>=1825):
        data_survival.loc[i, 'label'] = 1
    elif (data_survival.loc[i, 'OS'] == 0) & (data_survival.loc[i, 'OS.time'] >= 1825):
        data_survival.loc[i, 'label'] = 1
data_survival=data_survival.rename(columns={'xena_sample':'sample'})

sample_all=set(sample_mRNA_data['sampleID'])&set(sample_DNA_data['sampleID'])&set(sample_CNV_data['sampleID'])&set(sample_CNV_gene_data['sampleID'])
sample_3=list(data_survival.loc[pd.notnull(data_survival['label']),'sample'])
sample_3=list(sample_all&set(sample_3))
sample_id_3=sample_mRNA_data.loc[sample_mRNA_data.sampleID.isin(sample_3),['sample_id','sampleID']]
sample_id_3.index=range(len(sample_id_3))
for i in range(len(sample_id_3)):
    sample_id_3.loc[i,'label']=list(data_survival.loc[data_survival['sample']==sample_id_3.loc[i,'sampleID'],'label'])[0]
sample_id_3.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_id_label_3.txt',sep='\t',index=False)

sample_id_3=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_id_label_3.txt',sep='\t')
sample_id_3['y']=sample_id_3['label'].astype(int)
sample_id_3.index=range(len(sample_id_3))
rskf =RepeatedStratifiedKFold(n_splits=5, n_repeats=2,random_state=1)
j=1
for train_index,test_index in rskf.split(sample_id_3, np.array(sample_id_3['y']).astype(int)):
    sample_id_3.loc[train_index,'dataset_'+str(j)]='discovery'
    sample_id_3.loc[test_index, 'dataset_' + str(j)] = 'validation'
    j=j+1
sample_id_3.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival.txt',sep='\t',index=False)
