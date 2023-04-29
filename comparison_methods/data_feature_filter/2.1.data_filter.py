import pandas as pd
import numpy as np
import argparse


def get_data(type, feature_num_type, path, feature_path, sample):
    if type == 'count':
        data = pd.read_csv(path + '/RNA_all_TPM.txt', sep='\t')
        data = data.rename(columns={data.columns[0]: 'feature'})
        data=data.drop_duplicates()
        data = data.fillna(0)
        feature_select = list(pd.read_csv(feature_path +'filter_feature_'+feature_num_type+'_RNA_all_TPM.txt', sep='\t', header=None)[0])
        if 'nan' in feature_select:
            feature_select.remove('nan')
        data = data.set_index('feature')
        data_select_all = data.loc[feature_select, sample]

    elif type == 'methylation':
        m_type_list=['methylation_count','methylation_max','methylation_min','methylation_mean',
                     'promoter_methylation_count','promoter_methylation_max','promoter_methylation_min','promoter_methylation_mean']
        data_select_all=pd.DataFrame(columns=sample)
        for t in m_type_list:
            print(t)
            data=pd.read_csv(path + t+'.txt', sep='\t')
            data = data.rename(columns={data.columns[0]: 'feature'})
            data = data.drop_duplicates()
            data = data.fillna(0)
            feature_select = list(pd.read_csv(feature_path +'filter_feature_'+feature_num_type+'_'+t+'.txt', sep='\t', header=None)[0])
            if 'nan' in feature_select:
                feature_select.remove('nan')
            data = data.set_index('feature')
            data_select = data.loc[feature_select, sample]
            data_select.index = [i + '_'+t for i in feature_select]
            data_select_all = pd.concat([data_select_all,data_select], axis=0)

    elif type == 'CNV':
        c_type_list=['CNV_count','CNV_max','CNV_min','CNV_mean','CNV_gene_level']
        data_select_all=pd.DataFrame(columns=sample)
        for t in c_type_list:
            print(t)
            data=pd.read_csv(path + t+'.txt', sep='\t')
            data = data.rename(columns={data.columns[0]: 'feature'})
            data = data.drop_duplicates()
            data = data.fillna(0)
            feature_select = list(pd.read_csv(feature_path +'filter_feature_'+feature_num_type+'_'+t+'.txt', sep='\t', header=None)[0])
            if 'nan' in feature_select:
                feature_select.remove('nan')
            data = data.set_index('feature')
            data_select = data.loc[feature_select, sample]
            data_select.index = [i + '_'+t for i in feature_select]
            data_select_all = pd.concat([data_select_all,data_select], axis=0)

    return data_select_all


def main_data(path, feature_path, label_path, feature_num_type, save_path):
    label = pd.read_csv(label_path, sep='\t')
    sample = list(label['sample_id'])
    data_count = get_data('count',feature_num_type, path, feature_path, sample)
    data_methylation = get_data('methylation', feature_num_type, path, feature_path, sample)
    data_CNV = get_data('CNV', feature_num_type, path, feature_path, sample)
    data_all = pd.concat([data_count, data_methylation, data_CNV], axis=0)

    data_count.to_csv(save_path + '/' + str(feature_num_type.split('_')[1]) + '/data_count.txt', sep='\t')
    data_methylation.to_csv(save_path + '/' + str(feature_num_type.split('_')[1]) + '/data_methylation.txt', sep='\t')
    data_CNV.to_csv(save_path + '/' + str(feature_num_type.split('_')[1]) + '/data_CNV.txt', sep='\t')
    data_all.to_csv(save_path + '/' + str(feature_num_type.split('_')[1]) + '/data_all.txt', sep='\t')


def main(args):
    main_data(args.path, args.feature_path, args.label_path, args.feature_num_type, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--path', type=str, required=True,
                        help='path', dest='path')
    parser.add_argument('--feature_path', type=str, required=True,
                        help='feature_path', dest='feature_path')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--feature_num_type', type=str, required=True,
                        help='feature_num_type', dest='feature_num_type')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)







