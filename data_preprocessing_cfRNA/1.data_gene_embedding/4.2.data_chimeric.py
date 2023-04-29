import pandas as pd
import numpy as np

def get_feature(input_path,save_path):
    data=pd.read_csv(input_path+'/chimericRNA_rawdata.txt',sep='\t')
    data=data.fillna(0)
    sample=list(data.columns[1:])
    ID=pd.read_csv(input_path+'/chimeric_ID.txt',sep='\t')
    ID=ID.drop_duplicates()

    data_1=data.copy()
    data_1['ID']=data['ID'].map(lambda x:x.split('|')[2])
    data_2=data.copy()
    data_2['ID']=data['ID'].map(lambda x:x.split('|')[3])
    data_new=pd.concat([data_1[['ID']+sample],data_2[['ID']+sample]])
    data_new=pd.merge(data_new.rename(columns={'ID':'feature'}),ID,on='feature',how='left')
    data_new=data_new.set_index('gene_id')

    data_=data_new[sample]
    data_[data_>0]=1
    data_=data_.reset_index()
    data_count=data_[['gene_id']+sample].groupby('gene_id').sum()

    data_count=data_count.reset_index()
    data_count.to_csv(save_path+'/chimeric_count.txt',sep='\t',index=False)

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
