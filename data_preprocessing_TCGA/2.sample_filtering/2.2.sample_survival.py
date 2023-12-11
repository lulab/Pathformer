import pandas as pd
import numpy as np
import argparse

def get_label(survival_label_path,save_path):

    data_survival=pd.read_csv(survival_label_path,sep='\t')

    sample_mRNA_data=pd.read_csv(save_path+'/sample_mRNA_data_2.txt',sep='\t')
    sample_mRNA_data['sampleID']=sample_mRNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
    sample_DNA_data=pd.read_csv(save_path+'/sample_DNA_data_2.txt',sep='\t')
    sample_DNA_data['sampleID']=sample_DNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
    sample_CNV_data=pd.read_csv(save_path+'/sample_CNV_data_2.txt',sep='\t')
    sample_CNV_data['sampleID']=sample_CNV_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
    sample_CNV_gene_data=pd.read_csv(save_path+'/sample_CNV_gene_data_2.txt',sep='\t')
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
    sample_id_3.to_csv(save_path+'/sample_id_label_3.txt',sep='\t',index=False)

    sample_id_3=pd.read_csv(save_path+'/sample_id_label_3.txt',sep='\t')
    sample_id_3['y']=sample_id_3['label'].astype(int)
    sample_id_3.index=range(len(sample_id_3))
    sample_id_3.to_csv(save_path+'/sample_survival.txt',sep='\t',index=False)

def main(args):
    get_label(args.survival_label_path,args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--survival_label_path', type=str, required=True,
                        help='survival_label_path', dest='survival_label_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)
