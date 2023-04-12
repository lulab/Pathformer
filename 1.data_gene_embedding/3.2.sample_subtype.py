import pandas as pd
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/subtypes_R.txt',sep=',')

data_2_BRCA=data_2.loc[data_2['cancer.type']=='BRCA',:]
data_BRCA=pd.DataFrame(columns=['sample_id','subtype','sampleID'])
data_BRCA['sample_id']=data_2_BRCA['pan.samplesID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
data_BRCA['sampleID']=data_2_BRCA['pan.samplesID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
data_BRCA['subtype']=data_2_BRCA['Subtype_Selected'].map(lambda x:x.split('.')[1])
cancer='BRCA'
sample_mRNA_data = pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/TCGA_new/' + cancer + '/sample_id/sample_mRNA_data_2.txt',sep='\t')
sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
sample_DNA_data = pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/TCGA_new/' + cancer + '/sample_id/sample_DNA_data_2.txt',sep='\t')
sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
sample_CNV_data = pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/TCGA_new/' + cancer + '/sample_id/sample_CNV_data_2.txt',sep='\t')
sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
sample_CNV_gene_data = pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/TCGA_new/' + cancer + '/sample_id/sample_CNV_gene_data_2.txt',sep='\t')
sample_CNV_gene_data['sampleID'] = sample_CNV_gene_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
sample_all = set(sample_mRNA_data['sampleID']) & set(sample_DNA_data['sampleID']) & set(sample_CNV_data['sampleID']) & set(sample_CNV_gene_data['sampleID'])
data_BRCA=data_BRCA.loc[data_BRCA.sampleID.isin(sample_all)]
print('all',len(sample_all))
print('BRCA',len(data_BRCA))
data_BRCA.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_id_label_subtype.txt',sep='\t', index=False)

data_BRCA['y'] = data_BRCA['subtype'].astype('category').cat.codes
data_BRCA.index = range(len(data_BRCA))
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
j = 1
for train_index, test_index in rskf.split(data_BRCA, np.array(data_BRCA['y']).astype(int)):
    data_BRCA.loc[train_index, 'dataset_' + str(j)] = 'discovery'
    data_BRCA.loc[test_index, 'dataset_' + str(j)] = 'validation'
    j = j + 1
data_BRCA.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype.txt', sep='\t', index=False)
