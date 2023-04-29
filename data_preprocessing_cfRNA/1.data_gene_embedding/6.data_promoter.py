import pandas as pd
import numpy as np
from numba import jit


def get_feature(input_path,save_path):
    data=pd.read_csv(input_path+'/promoter_rowdata_TPM.txt',sep='\t')
    data=data.fillna(0)
    sample=list(data.columns[1:])
    data['gene_id']=data['promoter_id'].map(lambda x:x.split('|')[0])
    data[data==0]=np.nan

    #max
    data_max=data[['gene_id']+sample].groupby('gene_id').max()
    data_max=data_max.fillna(0)
    data_max=data_max.reset_index()
    data_max.to_csv(save_path+'/promoter_max.txt',sep='\t',index=False)

    #min
    data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
    data_min=data_min.fillna(0)
    data_min.to_csv(save_path+'/promoter_min.txt',sep='\t',index=False)

    #mean
    data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
    data_mean=data_mean.fillna(0)
    data_mean=data_mean.reset_index()
    data_mean.to_csv(save_path+'/promoter_mean.txt',sep='\t',index=False)

    #count
    data=data.fillna(0)
    data_=data[sample]
    data_[data_>0]=1
    data_['gene_id']=data['gene_id']
    data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
    data_count=data_count.fillna(0)
    data_count=data_count.reset_index()
    data_count.to_csv(save_path+'/promoter_count.txt',sep='\t',index=False)

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