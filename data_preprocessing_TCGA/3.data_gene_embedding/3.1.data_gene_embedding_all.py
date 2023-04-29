import pandas as pd
import numpy as np
import re
import argparse


def data_filter(rawdata_path,embedding_data_path,reference_path,sample_id_path,save_path,cancer):
    # RNA
    print('RNA')
    data_mRNA=pd.read_csv(rawdata_path+cancer+'.mRNA.csv',sep=',')
    sample_mRNA_data=pd.read_csv(sample_id_path+'/sample_mRNA_data_2.txt',sep='\t')
    data_mRNA=data_mRNA[['Unnamed: 0']+list(sample_mRNA_data['sample_old'])]
    data_mRNA.columns=['id']+list(sample_mRNA_data['sample_id'])
    data_mRNA['gene_id']=data_mRNA['id'].map(lambda x:x.split('.')[0])
    data_mRNA=data_mRNA[['gene_id']+list(sample_mRNA_data['sample_id'])].groupby('gene_id').sum()
    data_mRNA=data_mRNA.reset_index()
    data_mRNA=data_mRNA.drop_duplicates()
    data_mRNA.to_csv(save_path+'/mRNA_rawdata.txt',sep='\t',index=False)

    data_miRNA=pd.read_csv(rawdata_path+cancer+'.miRNA.csv',sep=',')
    sample_miRNA_data=pd.read_csv(sample_id_path+'/sample_miRNA_data_2.txt',sep='\t')
    data_miRNA=data_miRNA[['miRNA_ID']+list(sample_miRNA_data['sample_old'])]
    data_miRNA.columns=['miRNA_ID']+list(sample_miRNA_data['sample_id'])
    if len(set(sample_mRNA_data['sample_id'])-set(sample_miRNA_data['sample_id']))>0:
        data_miRNA[list(set(sample_mRNA_data['sample_id'])-set(sample_miRNA_data['sample_id']))]=np.nan

    id_data=pd.read_csv(reference_path+'/miRNA_id_new.txt',sep='\t')
    id_data.columns=['gene_id','miRNA_ID','name']
    id_data=id_data.drop_duplicates()
    data_miRNA_new=pd.merge(data_miRNA,id_data[['gene_id','miRNA_ID']],how='left',on='miRNA_ID')
    data_miRNA_new=data_miRNA_new.loc[pd.notnull(data_miRNA_new['gene_id']),:]
    data_miRNA_new=data_miRNA_new.drop_duplicates('gene_id')
    data_miRNA_new=data_miRNA_new[['gene_id']+list(sample_mRNA_data['sample_id'])]
    data_miRNA_new=data_miRNA_new.drop_duplicates()
    data_miRNA_new.to_csv(save_path+'/miRNA_rawdata.txt',sep='\t',index=False)

    data_all=pd.concat([data_mRNA,data_miRNA_new.loc[data_miRNA_new.gene_id.isin(list(set(data_miRNA_new['gene_id'])-set(data_mRNA['gene_id'])))]])
    data_all=data_all[['gene_id']+list(set(data_mRNA.columns[1:])&set(data_miRNA_new.columns[1:]))]
    data_all=data_all.drop_duplicates()
    data_all.to_csv(save_path+'/RNA_all_rawdata.txt',sep='\t',index=False)

    df = pd.read_csv(save_path+'/RNA_all_rawdata.txt',sep="\t")
    df=df.set_index('gene_id')
    sample=list(df.columns[1:])
    length_data=pd.read_csv(reference_path+'/gene.length.new.txt',sep='\t',header=None)
    length_data.columns=['gene_id','length']
    print("Done .")
    print("Calculate TPM ...")
    gene = list(set(df.index)&set(length_data['gene_id']))
    length_data =length_data.set_index('gene_id')
    length=length_data.loc[gene,:]
    df=df.loc[gene,:]
    lengthScaledDf = pd.DataFrame((df.values/length.values.reshape((-1,1))),index=df.index,columns=df.columns)
    data_1 = (1000000*lengthScaledDf.div(lengthScaledDf.sum(axis=0))).round(4)
    data_1=data_1.reset_index()
    data_1.to_csv(save_path+'/RNA_all_TPM.txt',sep='\t',index=False)

    #DNA
    print('DNA')
    sample_DNA_data=pd.read_csv(sample_id_path+'/sample_DNA_data_2.txt',sep='\t')
    data_methylation=pd.read_csv(rawdata_path+cancer+'.DNAmethy.csv',sep=',')
    data_methylation=data_methylation[['Unnamed: 0']+list(sample_DNA_data['sample_old'])]
    data_methylation.columns=['ID']+list(sample_DNA_data['sample_id'])
    data_methylation=data_methylation.drop_duplicates()
    data_methylation.to_csv(save_path+'/methylation_rawdata.txt',sep='\t',index=False)

    data_methylation_count=pd.read_csv(embedding_data_path+'/methylation_count.txt',sep='\t')
    data_methylation_count=data_methylation_count[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_methylation_count.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_methylation_count=data_methylation_count.drop_duplicates()
    data_methylation_count.to_csv(save_path+'/methylation_count.txt',sep='\t',index=False)

    data_methylation_max=pd.read_csv(embedding_data_path+'/methylation_max.txt',sep='\t')
    data_methylation_max=data_methylation_max[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_methylation_max.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_methylation_max=data_methylation_max.drop_duplicates()
    data_methylation_max.to_csv(save_path+'/methylation_max.txt',sep='\t',index=False)

    data_methylation_min=pd.read_csv(embedding_data_path+'/methylation_min.txt',sep='\t')
    data_methylation_min=data_methylation_min[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_methylation_min.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_methylation_min=data_methylation_min.drop_duplicates()
    data_methylation_min.to_csv(save_path+'/methylation_min.txt',sep='\t',index=False)

    data_methylation_mean=pd.read_csv(embedding_data_path+'/methylation_mean.txt',sep='\t')
    data_methylation_mean=data_methylation_mean[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_methylation_mean.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_methylation_mean=data_methylation_mean.drop_duplicates()
    data_methylation_mean.to_csv(save_path+'/methylation_mean.txt',sep='\t',index=False)

    data_promoter_methylation_count=pd.read_csv(embedding_data_path+'/promoter_methylation_count.txt',sep='\t')
    data_promoter_methylation_count=data_promoter_methylation_count[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_promoter_methylation_count.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_promoter_methylation_count=data_promoter_methylation_count.drop_duplicates()
    data_promoter_methylation_count.to_csv(save_path+'/promoter_methylation_count.txt',sep='\t',index=False)

    data_promoter_methylation_max=pd.read_csv(embedding_data_path+'/promoter_methylation_max.txt',sep='\t')
    data_promoter_methylation_max=data_promoter_methylation_max[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_promoter_methylation_max.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_promoter_methylation_max=data_promoter_methylation_max.drop_duplicates()
    data_promoter_methylation_max.to_csv(save_path+'/promoter_methylation_max.txt',sep='\t',index=False)

    data_promoter_methylation_min=pd.read_csv(embedding_data_path+'/promoter_methylation_min.txt',sep='\t')
    data_promoter_methylation_min=data_promoter_methylation_min[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_promoter_methylation_min.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_promoter_methylation_min=data_promoter_methylation_min.drop_duplicates()
    data_promoter_methylation_min.to_csv(save_path+'/promoter_methylation_min.txt',sep='\t',index=False)

    data_promoter_methylation_mean=pd.read_csv(embedding_data_path+'/promoter_methylation_mean.txt',sep='\t')
    data_promoter_methylation_mean=data_promoter_methylation_mean[['gene_id']+list(sample_DNA_data['sample_old'])]
    data_promoter_methylation_mean.columns=['gene_id']+list(sample_DNA_data['sample_id'])
    data_promoter_methylation_mean=data_promoter_methylation_mean.drop_duplicates()
    data_promoter_methylation_mean.to_csv(save_path+'/promoter_methylation_mean.txt',sep='\t',index=False)

    #CNV
    print('CNV')
    sample_CNV_data=pd.read_csv(sample_id_path+'/sample_CNV_data_2.txt',sep='\t')

    data_CNV=pd.read_csv(rawdata_path+cancer+'.CNV_masked_seg.csv',sep='\t')
    data_CNV=data_CNV[['ID']+list(sample_CNV_data['sample_old'])]
    data_CNV.columns=['ID']+list(sample_CNV_data['sample_id'])
    data_CNV=data_CNV.drop_duplicates()
    data_CNV.to_csv(save_path+'/CNV_masked_rawdata.txt',sep='\t',index=False)

    data_CNV_count=pd.read_csv(embedding_data_path+'/CNV_count.txt',sep='\t')
    data_CNV_count=data_CNV_count[['gene_id']+list(sample_CNV_data['sample_old'])]
    data_CNV_count.columns=['gene_id']+list(sample_CNV_data['sample_id'])
    data_CNV_count=data_CNV_count.drop_duplicates()
    data_CNV_count.to_csv(save_path+'/CNV_count.txt',sep='\t',index=False)

    data_CNV_max=pd.read_csv(embedding_data_path+'/CNV_max.txt',sep='\t')
    data_CNV_max=data_CNV_max[['gene_id']+list(sample_CNV_data['sample_old'])]
    data_CNV_max.columns=['gene_id']+list(sample_CNV_data['sample_id'])
    data_CNV_max=data_CNV_max.drop_duplicates()
    data_CNV_max.to_csv(save_path+'/CNV_max.txt',sep='\t',index=False)

    data_CNV_min=pd.read_csv(embedding_data_path+'/CNV_min.txt',sep='\t')
    data_CNV_min=data_CNV_min[['gene_id']+list(sample_CNV_data['sample_old'])]
    data_CNV_min.columns=['gene_id']+list(sample_CNV_data['sample_id'])
    data_CNV_min=data_CNV_min.drop_duplicates()
    data_CNV_min.to_csv(save_path+'/CNV_min.txt',sep='\t',index=False)

    data_CNV_mean=pd.read_csv(embedding_data_path+'/CNV_mean.txt',sep='\t')
    data_CNV_mean=data_CNV_mean[['gene_id']+list(sample_CNV_data['sample_old'])]
    data_CNV_mean.columns=['gene_id']+list(sample_CNV_data['sample_id'])
    data_CNV_mean=data_CNV_mean.drop_duplicates()
    data_CNV_mean.to_csv(save_path+'/CNV_mean.txt',sep='\t',index=False)

    #CNV
    print('CNV_gene')
    sample_CNV_gene_data=pd.read_csv(sample_id_path+'/sample_CNV_gene_data_2.txt',sep='\t')
    data_CNV=pd.read_csv(rawdata_path+cancer+'.CNV_gene_level.txt',sep='\t')
    data_CNV['Gene Symbol_new']=data_CNV['Gene Symbol'].map(lambda x:x.split('|')[0])
    CVN_gene_num=data_CNV['Gene Symbol_new'].value_counts()
    CVN_gene_num=CVN_gene_num.reset_index()
    CVN_gene=list(CVN_gene_num.loc[CVN_gene_num['Gene Symbol_new']>1,'index'])
    data_CNV=data_CNV.rename(columns={'Gene Symbol':'gene_name'})
    gtf_data=pd.read_csv(reference_path+'/gtf_name_103.txt',sep='\t')
    gtf_data_1=gtf_data.loc[gtf_data.gene_name.isin(CVN_gene)]
    gtf_data_1['gene_name']=gtf_data_1['gene_name']+'|chr'+gtf_data_1['chr']
    gtf_data_new=pd.concat([gtf_data,gtf_data_1])
    data_CNV_new=pd.merge(data_CNV,gtf_data_new[['gene_name','gene_id']],on='gene_name',how='left')
    data_CNV_new=data_CNV_new.loc[pd.notnull(data_CNV_new['gene_id'])]
    data_CNV_new=data_CNV_new[['gene_id']+list(sample_CNV_gene_data['sample_old'])]
    data_CNV_new.columns=['gene_id']+list(sample_CNV_gene_data['sample_id'])
    data_CNV_new=data_CNV_new.drop_duplicates()
    data_CNV_new.to_csv(save_path+'/CNV_gene_level.txt',sep='\t',index=False)

def main(args):

    data_filter(args.rawdata_path, args.embedding_data_path,args.reference_path,args.sample_id_path ,args.save_path, args.cancer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine learning module')
    parser.add_argument('--rawdata_path', type=str, required=True,
                        help='rawdata_path', dest='rawdata_path')
    parser.add_argument('--embedding_data_path', type=str, required=True,
                        help='embedding_data_path', dest='embedding_data_path')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='reference_path', dest='reference_path')
    parser.add_argument('--sample_id_path', type=str, required=True,
                        help='sample_id_path', dest='sample_id_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cancer', type=str, required=True,
                        help='cancer', dest='cancer')
    args = parser.parse_args()
    main(args)