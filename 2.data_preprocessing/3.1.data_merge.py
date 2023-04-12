import pandas as pd
import numpy as np
import argparse

def get_data(path,feature_type_path,label_path,save_path,feature_data):
    label=pd.read_csv(label_path,sep='\t')
    sample=list(label['sample_id'])
    feature_type=pd.read_csv(feature_type_path,sep='\t',header=None)
    feature_type['type']=feature_type[0].map(lambda x:x.split('_')[0])
    feature_type['type_name']=feature_type[0].map(lambda x:x.split('.')[0])

    data_all = np.zeros([len(feature_type), len(feature_data), len(sample)])

    for i in range(len(feature_type)):
        print(feature_type.loc[i,'type_name'])
        data=pd.read_csv(path+feature_type.loc[i,'type_name']+'.txt',sep='\t')
        data=data.rename(columns={data.columns[0]:'feature'})
        data = data.drop_duplicates()
        data=data.fillna(0)
        data=data[['feature']+sample]

        data['gene_id']=data['feature'].astype(str)
        data_=pd.DataFrame(columns=['gene_id']+sample)
        data_['gene_id']=list(set(feature_data)-set(data['gene_id']))
        data=pd.concat([data,data_])
        data=data.drop_duplicates('gene_id')
        data=data.set_index('gene_id')
        data_select=data.loc[feature_data,sample]
        data_select=data_select.fillna(0)

        data_all[i, :, :] = data_select.values
    data_all=data_all.transpose((2,1,0))
    np.save(file=save_path+'/data_all.npy',arr=data_all)


def main(args):
    feature_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/gene_all.txt',header=None)
    feature_data=list(feature_data[0])
    # get_data(path, feature_type_path, label_path, save_path, feature_data)
    get_data(args.path, args.feature_type_path, args.label_path, args.save_path, feature_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--path', type=str,required=True,
                        help='path',dest='path')
    parser.add_argument('--feature_type_path', type=str,required=True,
                        help='feature_type_path',dest='feature_type_path')
    parser.add_argument('--label_path', type=str,required=True,
                        help='label_path',dest='label_path')
    parser.add_argument('--save_path', type=str,required=True,
                        help='save_path',dest='save_path')
    args = parser.parse_args()
    main(args)







