import pandas as pd
import numpy as np
from numba import jit
import scipy.stats
import argparse

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

def get_feature(input_path,save_path):
    data=pd.read_csv(input_path+'/AS_rawdata.txt',sep='\t')
    data=data.fillna(0)
    sample=list(data.columns[1:])
    data['gene_id']=data['feature'].map(lambda x:x.split('|')[0]+'|'+x.split('|')[1])
    gene_id_list=set(data['gene_id'])
    data[data==0]=np.nan

    #max
    data_max=data[['gene_id']+sample].groupby('gene_id').max()
    data_max=data_max.fillna(0)
    data_max=data_max.reset_index()
    data_max.to_csv(save_path+'/splicing_max.txt',sep='\t',index=False)

    #min
    data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
    data_min=data_min.fillna(0)
    # data_min=data_min.reset_index()
    data_min.to_csv(save_path+'/splicing_min.txt',sep='\t',index=False)

    #mean
    data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
    data_mean=data_mean.fillna(0)
    data_mean=data_mean.reset_index()
    data_mean.to_csv(save_path+'/splicing_mean.txt',sep='\t',index=False)

    #count
    data=data.fillna(0)
    data_=data[sample]
    data_[data_>0]=1
    data_['gene_id']=data['gene_id']
    data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
    data_count=data_count.fillna(0)
    data_count=data_count.reset_index()
    data_count.to_csv(save_path+'/splicing_count.txt',sep='\t',index=False)


    data_entropy=data.loc[:,['gene_id']+sample].groupby('gene_id').apply(get_entropy)
    data_entropy=data_entropy.fillna(0)
    data_entropy=data_entropy.reset_index()
    data_entropy.to_csv(save_path+'/splicing_entropy.txt',sep='\t',index=False)


    data=pd.read_csv(save_path+'/splicing_max.txt',sep='\t')
    data['type']=data['gene_id'].map(lambda x:x.split('|')[0])
    data_A3SS=data.loc[data['type']=='A3SS']
    data_A3SS=data_A3SS[['gene_id']+sample]
    data_A5SS=data.loc[data['type']=='A5SS']
    data_A5SS=data_A5SS[['gene_id']+sample]
    data_MXE=data.loc[data['type']=='MXE']
    data_MXE=data_MXE[['gene_id']+sample]
    data_RI=data.loc[data['type']=='RI']
    data_RI=data_RI[['gene_id']+sample]
    data_SE=data.loc[data['type']=='SE']
    data_SE=data_SE[['gene_id']+sample]
    data_A3SS.to_csv(save_path+'/splicing_max_A3SS.txt',sep='\t',index=False)
    data_A5SS.to_csv(save_path+'/splicing_max_A5SS.txt',sep='\t',index=False)
    data_MXE.to_csv(save_path+'/splicing_max_MXE.txt',sep='\t',index=False)
    data_RI.to_csv(save_path+'/splicing_max_RI.txt',sep='\t',index=False)
    data_SE.to_csv(save_path+'/splicing_max_SE.txt',sep='\t',index=False)


    data=pd.read_csv(save_path+'/splicing_min.txt',sep='\t')
    data['type']=data['gene_id'].map(lambda x:x.split('|')[0])
    data_A3SS=data.loc[data['type']=='A3SS']
    data_A3SS=data_A3SS[['gene_id']+sample]
    data_A5SS=data.loc[data['type']=='A5SS']
    data_A5SS=data_A5SS[['gene_id']+sample]
    data_MXE=data.loc[data['type']=='MXE']
    data_MXE=data_MXE[['gene_id']+sample]
    data_RI=data.loc[data['type']=='RI']
    data_RI=data_RI[['gene_id']+sample]
    data_SE=data.loc[data['type']=='SE']
    data_SE=data_SE[['gene_id']+sample]
    data_A3SS.to_csv(save_path+'/splicing_min_A3SS.txt',sep='\t',index=False)
    data_A5SS.to_csv(save_path+'/splicing_min_A5SS.txt',sep='\t',index=False)
    data_MXE.to_csv(save_path+'/splicing_min_MXE.txt',sep='\t',index=False)
    data_RI.to_csv(save_path+'/splicing_min_RI.txt',sep='\t',index=False)
    data_SE.to_csv(save_path+'/splicing_min_SE.txt',sep='\t',index=False)

    data=pd.read_csv(save_path+'/splicing_mean.txt',sep='\t')
    data['type']=data['gene_id'].map(lambda x:x.split('|')[0])
    data_A3SS=data.loc[data['type']=='A3SS']
    data_A3SS=data_A3SS[['gene_id']+sample]
    data_A5SS=data.loc[data['type']=='A5SS']
    data_A5SS=data_A5SS[['gene_id']+sample]
    data_MXE=data.loc[data['type']=='MXE']
    data_MXE=data_MXE[['gene_id']+sample]
    data_RI=data.loc[data['type']=='RI']
    data_RI=data_RI[['gene_id']+sample]
    data_SE=data.loc[data['type']=='SE']
    data_SE=data_SE[['gene_id']+sample]
    data_A3SS.to_csv(save_path+'/splicing_mean_A3SS.txt',sep='\t',index=False)
    data_A5SS.to_csv(save_path+'/splicing_mean_A5SS.txt',sep='\t',index=False)
    data_MXE.to_csv(save_path+'/splicing_mean_MXE.txt',sep='\t',index=False)
    data_RI.to_csv(save_path+'/splicing_mean_RI.txt',sep='\t',index=False)
    data_SE.to_csv(save_path+'/splicing_mean_SE.txt',sep='\t',index=False)


    data=pd.read_csv(save_path+'/splicing_count.txt',sep='\t')
    data['type']=data['gene_id'].map(lambda x:x.split('|')[0])
    data_A3SS=data.loc[data['type']=='A3SS']
    data_A3SS=data_A3SS[['gene_id']+sample]
    data_A5SS=data.loc[data['type']=='A5SS']
    data_A5SS=data_A5SS[['gene_id']+sample]
    data_MXE=data.loc[data['type']=='MXE']
    data_MXE=data_MXE[['gene_id']+sample]
    data_RI=data.loc[data['type']=='RI']
    data_RI=data_RI[['gene_id']+sample]
    data_SE=data.loc[data['type']=='SE']
    data_SE=data_SE[['gene_id']+sample]
    data_A3SS.to_csv(save_path+'/splicing_count_A3SS.txt',sep='\t',index=False)
    data_A5SS.to_csv(save_path+'/splicing_count_A5SS.txt',sep='\t',index=False)
    data_MXE.to_csv(save_path+'/splicing_count_MXE.txt',sep='\t',index=False)
    data_RI.to_csv(save_path+'/splicing_count_RI.txt',sep='\t',index=False)
    data_SE.to_csv(save_path+'/splicing_count_SE.txt',sep='\t',index=False)

    data=pd.read_csv(save_path+'/splicing_entropy.txt',sep='\t')
    data['type']=data['gene_id'].map(lambda x:x.split('|')[0])
    data_A3SS=data.loc[data['type']=='A3SS']
    data_A3SS=data_A3SS[['gene_id']+sample]
    data_A5SS=data.loc[data['type']=='A5SS']
    data_A5SS=data_A5SS[['gene_id']+sample]
    data_MXE=data.loc[data['type']=='MXE']
    data_MXE=data_MXE[['gene_id']+sample]
    data_RI=data.loc[data['type']=='RI']
    data_RI=data_RI[['gene_id']+sample]
    data_SE=data.loc[data['type']=='SE']
    data_SE=data_SE[['gene_id']+sample]
    data_A3SS.to_csv(save_path+'/splicing_entropy_A3SS.txt',sep='\t',index=False)
    data_A5SS.to_csv(save_path+'/splicing_entropy_A5SS.txt',sep='\t',index=False)
    data_MXE.to_csv(save_path+'/splicing_entropy_MXE.txt',sep='\t',index=False)
    data_RI.to_csv(save_path+'/splicing_entropy_RI.txt',sep='\t',index=False)
    data_SE.to_csv(save_path+'/splicing_entropy_SE.txt',sep='\t',index=False)

def main(args):
    get_feature(args.input_path, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--input_path', type=str, required=True,
        help='input_path',dest='input_path')
    parser.add_argument('--save_path', type=str, required=True,
        help='save_path',dest='save_path')
    args = parser.parse_args()
    main(args)
