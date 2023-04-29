import pandas as pd
import numpy as np
import re
import scipy.stats
import argparse

def get_data(rawdata_path,save_path,cancer):
    data=pd.read_csv(rawdata_path+'/'+cancer+'.DNAmethy.csv',sep=',')
    sample=data.columns[1:]
    data.columns=['ID']+list(sample)

    ID_gene=pd.read_csv(save_path+'/DNA_methylation_geneid.txt',sep='\t')
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
    data_max.to_csv(save_path+'/methylation_max.txt',sep='\t',index=False)
    del data_max
    #min
    data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
    data_min=data_min.fillna(0)
    # data_min=data_min.reset_index()
    data_min.to_csv(save_path+'/methylation_min.txt',sep='\t',index=False)
    del data_min
    #mean
    data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
    data_mean=data_mean.fillna(0)
    data_mean=data_mean.reset_index()
    data_mean.to_csv(save_path+'/methylation_mean.txt',sep='\t',index=False)
    del data_mean
    #count
    data=data.fillna(0)
    data_=data[sample].copy()
    data_[data_>0]=1
    data_['gene_id']=data['gene_id']
    data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
    data_count=data_count.fillna(0)
    data_count=data_count.reset_index()
    data_count.to_csv(save_path+'/methylation_count.txt',sep='\t',index=False)

def main(args):
    get_data(args.rawdata_path,args.save_path,args.cancer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--rawdata_path', type=str, required=True,
                        help='rawdata_path', dest='rawdata_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cancer', type=str, required=True,
                        help='cancer', dest='cancer')
    args = parser.parse_args()
    main(args)
