import pandas as pd
import numpy as np
import argparse

def get_data(path,label_path,reference_path,save_path):
    feature_data = pd.read_csv(reference_path+'/gene_all.txt',header=None)
    feature_data = list(feature_data[0])
    label=pd.read_csv(label_path,sep='\t')
    sample=list(label['sample_id'])
    count_type=['RNA_all_TPM']
    m_type_list = ['methylation_count', 'methylation_max', 'methylation_min', 'methylation_mean',
                   'promoter_methylation_count', 'promoter_methylation_max', 'promoter_methylation_min',
                   'promoter_methylation_mean']
    c_type_list = ['CNV_count', 'CNV_max', 'CNV_min', 'CNV_mean', 'CNV_gene_level']

    data_count_all=pd.DataFrame(columns=sample)

    for i in range(len(count_type)):
        data=pd.read_csv(path+count_type[i]+'.txt',sep='\t')
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
        data_count_all=pd.concat([data_count_all,data_select])

    data_methylation_all=pd.DataFrame(columns=sample)

    for i in range(len(m_type_list)):
        data=pd.read_csv(path+m_type_list[i]+'.txt',sep='\t')
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
        data_methylation_all=pd.concat([data_methylation_all,data_select])

    data_CNV_all=pd.DataFrame(columns=sample)

    for i in range(len(c_type_list)):
        data=pd.read_csv(path+c_type_list[i]+'.txt',sep='\t')
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
        data_CNV_all=pd.concat([data_CNV_all,data_select])

    data_count_all.to_csv(save_path+'data_count_all.txt',sep='\t')
    data_methylation_all.to_csv(save_path + 'data_methylation_all.txt', sep='\t')
    data_CNV_all.to_csv(save_path + 'data_CNV_all.txt', sep='\t')



def main(args):
    get_data(args.path, args.label_path, args.reference_path,args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--path', type=str, required=True,
                        help='path', dest='path')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='reference_path', dest='reference_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)







