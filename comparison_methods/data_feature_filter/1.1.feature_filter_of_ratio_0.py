import pandas as pd
import numpy as np
import argparse
from pandas.api.types import is_string_dtype

def get_ratio(data,nc,n=0):
    feature = list(data.index)
    data_NC = data.loc[feature,nc]
    ratio = pd.DataFrame(columns=['feature', 'ratio_all', 'ratio'])
    ratio_all = []
    ratio_NC = []
    j = 0
    # n = 0
    for i in feature:
        if j%1000==0:
            print(j)
        j=j+1
        X_all = np.array(data.loc[i,:])
        X_all_ = X_all[X_all>n]
        ratio_all.append(len(X_all_)/len(X_all))
        X_NC = np.array(data_NC.loc[i,:])
        # print(X_NC)
        X_NC_ = X_NC[X_NC>n]
        ratio_NC.append(len(X_NC_)/len(X_NC))

    ratio['feature']=feature
    ratio['ratio_all']=ratio_all
    ratio['ratio']=ratio_NC
    return ratio



def get_feature(feature_path,path_label,dataset,p,savepath,feature_type):
    data_1=pd.read_csv(feature_path,sep='\t')
    if is_string_dtype(data_1[data_1.columns[0]]):
        data_1 = data_1.rename(columns={data_1.columns[0]: 'feature'})
    else:
        data_1 = data_1.reset_index()
        data_1 = data_1.rename(columns={data_1.columns[0]: 'feature'})
    data_1=data_1.fillna(0)
    sample=pd.read_csv(path_label,sep='\t')
    sample_train_test=sample.loc[(sample['dataset_'+str(dataset)]=='discovery'),'sample_id']
    sample_validation=sample.loc[(sample['dataset_'+str(dataset)]=='validation'),'sample_id']
    data_1_train=data_1[['feature']+list(sample_train_test)]

    ratio_all=pd.DataFrame(columns=['feature'])
    ratio_all['feature']=data_1_train['feature']
    for type in list(set(sample['y'])):
        sample_type = list(set(sample.loc[(sample['y']==type),'sample_id'])&set(sample_train_test))
        data_2 = data_1_train[['feature']+list(sample_type)]
        data_2 = data_2.set_index('feature')
        ratio=get_ratio(data_2,sample_type,n=0)
        ratio_all[type]=ratio['ratio']
    ratio_all.to_csv(savepath+'/ratio_'+feature_type+'.txt',sep='\t',index=False)
    ratio_filiter=ratio.loc[(ratio_all[list(set(sample['y']))]>p).sum(axis=1)>0,:]
    pd.DataFrame(ratio_filiter['feature']).to_csv(savepath+'/filter_feature_'+feature_type+'.txt',sep='\t',index=False,header=False)

def main(args):
    get_feature(args.feature_path, args.path_label, args.dataset, args.p, args.savepath,args.feature_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--feature_path', type=str, required=True,
        help='feature_path',dest='feature_path')
    parser.add_argument('--path_label', type=str, required=True,
        help='path_label',dest='path_label')
    parser.add_argument('--dataset', type=int, required=True,
        help='dataset',dest='dataset')
    parser.add_argument('--p', type=float,required=True,
        help='p',dest='p')
    parser.add_argument('--savepath', type=str,required=True,
        help='savepath',dest='savepath')
    parser.add_argument('--feature_type', type=str,required=True,
        help='feature_type',dest='feature_type')
    args = parser.parse_args()
    main(args)


