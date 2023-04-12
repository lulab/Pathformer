import pandas as pd
import numpy as np
import re

data_ID=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/BRCA.DNAmethy.csv',sep=',',usecols=[0])
data_ID.columns=['ID']

CPG_id=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/HM450.hg38.manifest.gencode.v36.tsv',sep='\t')
CPG_id=CPG_id.rename(columns=['probeID':'ID'])
data_ID=pd.merge(data_ID,CPG_id[['ID','CpG_beg','CpG_end','CpG_chrm','transcriptIDs','probe_strand']],on='ID',how='left')
data_ID=data_ID.loc[pd.notnull(data_ID['transcriptIDs'])]
data_ID['transcript_id']=data_ID['transcriptIDs'].str.split(';')
data_ID_new=data_ID.explode('transcript_id')
data_ID_new['transcript_id']=data_ID_new['transcript_id'].map(lambda x:x.split('.')[0])

gtf_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Homo_sapiens.GRCh38.91.chr.gtf',sep='\t',skiprows = lambda x: x in [0,1,2,3,4],header=None)
gtf_data=gtf_data.loc[gtf_data.iloc[:,2]=='transcript',:]
gtf_data_new = pd.DataFrame(columns=['gene_id','transcript_id','chr','strat','end','strand'])
gtf_data_new['transcript_id'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('transcript_id ".*?"',x)[0].split('"')[1])
gtf_data_new['gene_id'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_id ".*?"',x)[0].split('"')[1])
gtf_data_new['gene_type'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_biotype ".*?"',x)[0].split('"')[1] if 'gene_biotype' in x else np.nan)
gtf_data_new['chr'] = gtf_data.iloc[:,0]
gtf_data_new['strat']=gtf_data.iloc[:,3].astype('int')
gtf_data_new['end']=gtf_data.iloc[:,4].astype('int')
gtf_data_new['strand']=gtf_data.iloc[:,6]
gtf_data_new = gtf_data_new.drop_duplicates()
gtf_data_new.index = range(len(gtf_data_new))

data_ID_new=pd.merge(data_ID_new,gtf_data_new[['gene_id','transcript_id']],on='transcript_id',how='left')
data_ID_new=data_ID_new.loc[pd.notnull(data_ID_new['gene_id'])]
data_ID_new=data_ID_new.drop_duplicates(['ID','gene_id'])
data_ID_new.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/DNA_methylation_geneid.txt',sep='\t',index=False)

promoter_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/promoter.bed',sep='\t',header=None)
promoter_data.columns=['chrm','strat','end','gene_id','dot','strand']
promoter_data['gene_id']=promoter_data['gene_id'].map(lambda x:x.split('_')[1].split('.')[0])

data_ID_new_new=pd.merge(data_ID_new,promoter_data[['gene_id','strat','end']],on='gene_id',how='left')
data_ID_promoter=data_ID_new_new.loc[(data_ID_new_new['CpG_end']>=data_ID_new_new['strat'])&(data_ID_new_new['CpG_end']<=data_ID_new_new['end']),:]
data_ID_promoter=data_ID_promoter[['ID', 'CpG_beg', 'CpG_end', 'CpG_chrm', 'transcriptIDs','transcript_id', 'gene_id']]
data_ID_promoter=data_ID_promoter.drop_duplicates(['ID','gene_id'])
data_ID_promoter.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/DNA_methylation_promoterid.txt',sep='\t',index=False)
