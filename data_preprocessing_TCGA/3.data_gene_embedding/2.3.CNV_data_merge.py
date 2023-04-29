import pandas as pd
import numpy as np
import re
import argparse

def get_data(save_path):
    data_up_max=pd.read_csv(save_path+'/data_up_max.txt',sep='\t')
    data_up_min=pd.read_csv(save_path+'/data_up_min.txt',sep='\t')
    data_down_max=pd.read_csv(save_path+'/data_down_max.txt',sep='\t')
    data_down_min=pd.read_csv(save_path+'/data_down_min.txt',sep='\t')

    data_up_max=data_up_max.set_index('gene_id')
    data_up_max=data_up_max.fillna(0)
    data_up_min=data_up_min.set_index('gene_id')
    data_up_min=data_up_min.fillna(0)
    data_down_max=data_down_max.set_index('gene_id')
    data_down_max=data_down_max.fillna(0)
    data_down_min=data_down_min.set_index('gene_id')
    data_down_min=data_down_min.fillna(0)

    data_up_max[data_down_min.abs()>data_up_max.abs()]=data_down_min
    data_up_min[data_down_max.abs()<data_up_min.abs()]=data_down_max

    data_up_max.to_csv(save_path+'/CNV_max.txt',sep='\t')
    data_up_min.to_csv(save_path+'/CNV_min.txt',sep='\t')


    data_up_sum=pd.read_csv(save_path+'/data_up_sum.txt',sep='\t')
    data_up_mean=pd.read_csv(save_path+'/data_up_mean.txt',sep='\t')
    data_down_sum=pd.read_csv(save_path+'/data_down_sum.txt',sep='\t')
    data_down_mean=pd.read_csv(save_path+'/data_down_mean.txt',sep='\t')

    data_up_sum=data_up_sum.set_index('gene_id')
    data_up_sum=data_up_sum.fillna(0)
    data_up_mean=data_up_mean.set_index('gene_id')
    data_up_mean=data_up_mean.fillna(0)
    data_down_sum=data_down_sum.set_index('gene_id')
    data_down_sum=data_down_sum.fillna(0)
    data_down_mean=data_down_mean.set_index('gene_id')
    data_down_mean=data_down_mean.fillna(0)

    data_up_mean[data_down_sum.abs()>data_up_sum.abs()]=data_down_mean
    data_up_mean.to_csv(save_path+'/CNV_mean.txt',sep='\t')

def main(args):
    get_data(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)