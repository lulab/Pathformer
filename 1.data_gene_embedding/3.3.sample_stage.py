import pandas as pd

print(cancer)
data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix',sep='\t')

sample_mRNA_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_mRNA_data_2.txt',sep='\t')
sample_mRNA_data['sampleID']=sample_mRNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_DNA_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_DNA_data_2.txt',sep='\t')
sample_DNA_data['sampleID']=sample_DNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_CNV_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_CNV_data_2.txt',sep='\t')
sample_CNV_data['sampleID']=sample_CNV_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
sample_CNV_gene_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_CNV_gene_data_2.txt',sep='\t')
sample_CNV_gene_data['sampleID']=sample_CNV_gene_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])

sample_all=set(sample_mRNA_data['sampleID'])&set(sample_DNA_data['sampleID'])&set(sample_CNV_data['sampleID'])&set(sample_CNV_gene_data['sampleID'])
if 'pathologic_stage' in list(data.columns):
    print('pathologic_stage')
    sample_all=set(sample_mRNA_data['sampleID'])&set(sample_DNA_data['sampleID'])&set(sample_CNV_data['sampleID'])&set(sample_CNV_gene_data['sampleID'])
    sample_2=list(data.loc[pd.notnull(data['pathologic_stage']),'sampleID'])
    sample_2=list(sample_all&set(sample_2))
    sample_id_2=sample_mRNA_data.loc[sample_mRNA_data.sampleID.isin(sample_2),['sample_id','sampleID']]
    sample_id_2.index=range(len(sample_id_2))
    for i in range(len(sample_id_2)):
        sample_id_2.loc[i,'stage']=list(data.loc[data['sampleID']==sample_id_2.loc[i,'sampleID'],'pathologic_stage'])[0]
    sample_id_2['label']=sample_id_2['stage'].replace('Stage IIA',0).replace('Stage IIB',0).replace('Stage IIIA',1).replace('Stage IA',0).replace('Stage I',0)\
        .replace('Stage IIIC',1).replace('Stage IIIB',1).replace('Stage IV',1).replace('Stage II',0).replace('Stage IB',0).replace('Stage III',1)
    sample_id_2.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_id_label_2.txt',sep='\t',index=False)
else:
    print('miss')

sample_id_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_id_label_2.txt',sep='\t')
sample_id_2['label']=sample_id_2['label'].replace('Stage IIC',0).replace('Stage IVA',1).replace('Stage IVB',1).replace('Stage IVC',1)
sample_id_2=sample_id_2.loc[(sample_id_2['label']=='0')|(sample_id_2['label']=='1')|(sample_id_2['label']==0)|(sample_id_2['label']==1)]

sample_id_2['y']=sample_id_2['label'].astype(int)
sample_id_2.index=range(len(sample_id_2))
rskf =RepeatedStratifiedKFold(n_splits=5, n_repeats=2,random_state=1)
j=1
for train_index,test_index in rskf.split(sample_id_2, np.array(sample_id_2['y']).astype(int)):
    sample_id_2.loc[train_index,'dataset_'+str(j)]='discovery'
    sample_id_2.loc[test_index, 'dataset_' + str(j)] = 'validation'
    j=j+1
sample_id_2.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage.txt',sep='\t',index=False)


