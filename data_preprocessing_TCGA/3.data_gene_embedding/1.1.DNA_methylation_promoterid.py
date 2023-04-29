import pandas as pd
import numpy as np
import re
import argparse

def get_data(reference_path,rawdata_path,save_path,cancer):
    data_ID=pd.read_csv(rawdata_path+'/'+cancer+'.DNAmethy.csv',sep=',',usecols=[0])
    data_ID.columns=['ID']

    CPG_id=pd.read_csv(reference_path+'/HM450.hg38.manifest.gencode.v36.tsv',sep='\t')
    CPG_id=CPG_id.rename(columns={'probeID':'ID'})
    data_ID=pd.merge(data_ID,CPG_id[['ID','CpG_beg','CpG_end','CpG_chrm','transcriptIDs','probe_strand']],on='ID',how='left')
    data_ID=data_ID.loc[pd.notnull(data_ID['transcriptIDs'])]
    data_ID['transcript_id']=data_ID['transcriptIDs'].str.split(';')
    data_ID_new=data_ID.explode('transcript_id')
    data_ID_new['transcript_id']=data_ID_new['transcript_id'].map(lambda x:x.split('.')[0])

    gtf_data=pd.read_csv(reference_path+'/Homo_sapiens.GRCh38.91.chr.gtf',sep='\t',skiprows = lambda x: x in [0,1,2,3,4],header=None)
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
    data_ID_new.to_csv(save_path+'/DNA_methylation_geneid.txt',sep='\t',index=False)

    promoter_data=pd.read_csv(reference_path+'/promoter.bed',sep='\t',header=None)
    promoter_data.columns=['chrm','strat','end','gene_id','dot','strand']
    promoter_data['gene_id']=promoter_data['gene_id'].map(lambda x:x.split('_')[1].split('.')[0])

    data_ID_new_new=pd.merge(data_ID_new,promoter_data[['gene_id','strat','end']],on='gene_id',how='left')
    data_ID_promoter=data_ID_new_new.loc[(data_ID_new_new['CpG_end']>=data_ID_new_new['strat'])&(data_ID_new_new['CpG_end']<=data_ID_new_new['end']),:]
    data_ID_promoter=data_ID_promoter[['ID', 'CpG_beg', 'CpG_end', 'CpG_chrm', 'transcriptIDs','transcript_id', 'gene_id']]
    data_ID_promoter=data_ID_promoter.drop_duplicates(['ID','gene_id'])
    data_ID_promoter.to_csv(save_path+'/DNA_methylation_promoterid.txt',sep='\t',index=False)

def main(args):
    get_data(args.reference_path,args.rawdata_path,args.save_path,args.cancer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='reference_path', dest='reference_path')
    parser.add_argument('--rawdata_path', type=str, required=True,
                        help='rawdata_path', dest='rawdata_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cancer', type=str, required=True,
                        help='cancer', dest='cancer')
    args = parser.parse_args()
    main(args)