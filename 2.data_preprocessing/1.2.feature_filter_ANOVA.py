import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse
from pandas.api.types import is_string_dtype


def BH(p_vulae):
    p_BH = multipletests(p_vulae, method='fdr_bh')[1]
    return p_BH

def get_feature(data,num=1000):
    if num >0 :
        data_select=data.loc[data['FDR']<=0.05,:]
        if len(data_select)<200:
            data_select = data.loc[data['P'] <= 0.05, :]
            if len(data_select)<20:
                data_select=data
        if len(data_select) > num:
            data_select=data_select.sort_values('FDR')
            feature_select=list(data_select['feature'])[:num]
        else:
            feature_select = list(data_select['feature'])
    else:
        data_select=data.loc[data['FDR']<=0.05,:]
        if len(data_select)<200:
            data_select = data.loc[data['P'] <= 0.05, :]
            if len(data_select) < 20:
                data_select = data
        feature_select= list(data_select['feature'])
    if 'nan' in feature_select:
        feature_select.remove('nan')
    return feature_select

def get_feature_all(feature_path, path_label, dataset_num, scaler, savepath, feature_type):
    feature_0_data=list(pd.read_csv(savepath+'filter_feature_'+feature_type+'.txt',sep='\t',header=None)[0])
    feature_0_data=list(set(feature_0_data))
    data_feature = pd.read_csv(feature_path, sep='\t')
    if is_string_dtype(data_feature[data_feature.columns[0]]):
        data_feature = data_feature.rename(columns={data_feature.columns[0]: 'feature'})
    else:
        data_feature = data_feature.reset_index()
        data_feature = data_feature.rename(columns={data_feature.columns[0]: 'feature'})
    data_feature = data_feature.fillna(0)
    data_feature=data_feature.drop_duplicates()

    data_label = pd.read_csv(path_label, sep='\t')
    sample_train = list(data_label.loc[data_label['dataset_' + str(dataset_num)] != 'validation', 'sample_id'])
    data_label = data_label.set_index('sample_id')

    y_train = np.array(data_label.loc[sample_train, 'y'])
    y_train = y_train.astype(int)

    data_train = data_feature[[data_feature.columns[0]] + sample_train]
    data_train=data_train.set_index(data_feature.columns[0])
    data_train=data_train.loc[feature_0_data,:]
    if scaler == 1:
        X = MinMaxScaler().fit_transform(np.array(data_train[sample_train]).T)
        data_train[sample_train] = X.T

    ######test FDR#########
    F, p = f_classif(data_train[sample_train].T, y_train)
    FDR = BH(p)
    data_ANOVA = pd.DataFrame(columns={'feature', 'FDR', 'P'})
    data_ANOVA['feature'] = list(data_train.index)
    data_ANOVA['FDR'] = FDR
    data_ANOVA['P'] = p
    data_ANOVA = data_ANOVA.sort_values('FDR')
    # print(len(data_ANOVA))
    # print(len(data_ANOVA.loc[data_ANOVA['FDR']<0.05]))
    data_ANOVA.to_csv(savepath + '/diff_ANOVA_'+feature_type+'.txt', sep='\t', index=False)
    feature_anova_100 = get_feature(data_ANOVA, num=100)
    feature_anova_500 = get_feature(data_ANOVA, num=500)
    feature_anova_1000 = get_feature(data_ANOVA, num=1000)
    feature_anova_all=get_feature(data_ANOVA, num=-1)
    pd.DataFrame(feature_anova_100).to_csv(savepath+'/filter_feature_ANOVA_100_'+feature_type+'.txt',sep='\t',index=False,header=False)
    pd.DataFrame(feature_anova_500).to_csv(savepath+'/filter_feature_ANOVA_500_'+feature_type+'.txt',sep='\t',index=False,header=False)
    pd.DataFrame(feature_anova_1000).to_csv(savepath+'/filter_feature_ANOVA_1000_'+feature_type+'.txt',sep='\t',index=False,header=False)
    pd.DataFrame(feature_anova_all).to_csv(savepath+'/filter_feature_ANOVA_all_'+feature_type+'.txt',sep='\t',index=False,header=False)

    
def main(args):

    get_feature_all(args.feature_path, args.path_label, args.dataset_num, args.scaler, args.savepath, args.feature_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--feature_path', type=str, required=True,
        help='feature_path',dest='feature_path')
    parser.add_argument('--path_label', type=str, required=True,
        help='path_label',dest='path_label')
    parser.add_argument('--dataset_num', type=int, required=True,
        help='dataset_num',dest='dataset_num')
    parser.add_argument('--scaler', type=int,required=True,
        help='scaler',dest='scaler')
    parser.add_argument('--savepath', type=str,required=True,
        help='savepath',dest='savepath')
    parser.add_argument('--feature_type', type=str,required=True,
        help='feature_type',dest='feature_type')
    args = parser.parse_args()
    main(args)
