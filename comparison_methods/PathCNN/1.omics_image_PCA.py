import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_pac(dataset,label_path,data_path,save_path):
    pathway_data=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/pathway_PathCNN.txt',sep='\t')
    label=pd.read_csv(label_path,sep='\t')
    sample_train=list(label.loc[label['dataset_'+str(dataset)+'_new']=='train','sample_id'])
    sample_validation=list(label.loc[label['dataset_'+str(dataset)+'_new']=='validation','sample_id'])
    sample_test=list(label.loc[label['dataset_'+str(dataset)+'_new']=='test','sample_id'])

    data_count=pd.read_csv(data_path+'/data_count_all.txt',sep='\t')
    data_count=data_count.set_index(data_count.columns[0])
    data_methylation=pd.read_csv(data_path+'/data_methylation_all.txt',sep='\t')
    data_methylation=data_methylation.set_index(data_methylation.columns[0])
    data_CNV=pd.read_csv(data_path+'/data_CNV_all.txt',sep='\t')
    data_CNV=data_CNV.set_index(data_CNV.columns[0])

    data_count_train_all=pd.DataFrame(columns=sample_train)
    data_count_validation_all=pd.DataFrame(columns=sample_validation)
    data_count_test_all=pd.DataFrame(columns=sample_test)
    data_methylation_train_all=pd.DataFrame(columns=sample_train)
    data_methylation_validation_all=pd.DataFrame(columns=sample_validation)
    data_methylation_test_all=pd.DataFrame(columns=sample_test)
    data_CNV_train_all=pd.DataFrame(columns=sample_train)
    data_CNV_validation_all=pd.DataFrame(columns=sample_validation)
    data_CNV_test_all=pd.DataFrame(columns=sample_test)


    for i in range(len(pathway_data)):
        print(i)
        gene_list=pathway_data.loc[i,'gene_id_list'].split(',')
        data_count_train=data_count.loc[gene_list,sample_train]
        data_count_validation = data_count.loc[gene_list, sample_validation]
        data_count_test = data_count.loc[gene_list, sample_test]
        data_methylation_train=data_methylation.loc[gene_list,sample_train]
        data_methylation_validation = data_methylation.loc[gene_list, sample_validation]
        data_methylation_test = data_methylation.loc[gene_list, sample_test]
        data_CNV_train=data_CNV.loc[gene_list,sample_train]
        data_CNV_validation = data_CNV.loc[gene_list, sample_validation]
        data_CNV_test = data_CNV.loc[gene_list, sample_test]

        scaler_count = MinMaxScaler()
        scaler_methylation = MinMaxScaler()
        scaler_CNV = MinMaxScaler()
        scaler_count.fit(np.array(data_count_train).T)
        scaler_methylation.fit(np.array(data_methylation_train).T)
        scaler_CNV.fit(np.array(data_CNV_train).T)

        data_count_train_scale=scaler_count.transform(np.array(data_count_train).T).T
        data_count_validation_scale = scaler_count.transform(np.array(data_count_validation).T).T
        data_count_test_scale = scaler_count.transform(np.array(data_count_test).T).T

        data_methylation_train_scale=scaler_methylation.transform(np.array(data_methylation_train).T).T
        data_methylation_validation_scale = scaler_methylation.transform(np.array(data_methylation_validation).T).T
        data_methylation_test_scale = scaler_methylation.transform(np.array(data_methylation_test).T).T

        data_CNV_train_scale=scaler_CNV.transform(np.array(data_CNV_train).T).T
        data_CNV_validation_scale = scaler_CNV.transform(np.array(data_CNV_validation).T).T
        data_CNV_test_scale = scaler_CNV.transform(np.array(data_CNV_test).T).T


        pca_count = PCA(n_components=5)
        pca_methylation = PCA(n_components=5)
        pca_CNV = PCA(n_components=5)
        pca_count.fit(np.array(data_count_train_scale).T)
        pca_methylation.fit(np.array(data_methylation_train_scale).T)
        pca_CNV.fit(np.array(data_CNV_train_scale).T)

        data_count_train_new=pca_count.transform(np.array(data_count_train_scale).T).T
        data_count_train_new=pd.DataFrame(data_count_train_new)
        data_count_train_new.columns=sample_train
        data_count_validation_new = pca_count.transform(np.array(data_count_validation_scale).T).T
        data_count_validation_new=pd.DataFrame(data_count_validation_new)
        data_count_validation_new.columns=sample_validation
        data_count_test_new = pca_count.transform(np.array(data_count_test_scale).T).T
        data_count_test_new=pd.DataFrame(data_count_test_new)
        data_count_test_new.columns=sample_test


        data_methylation_train_new=pca_methylation.transform(np.array(data_methylation_train_scale).T).T
        data_methylation_train_new=pd.DataFrame(data_methylation_train_new)
        data_methylation_train_new.columns=sample_train
        data_methylation_validation_new = pca_methylation.transform(np.array(data_methylation_validation_scale).T).T
        data_methylation_validation_new=pd.DataFrame(data_methylation_validation_new)
        data_methylation_validation_new.columns=sample_validation
        data_methylation_test_new = pca_methylation.transform(np.array(data_methylation_test_scale).T).T
        data_methylation_test_new=pd.DataFrame(data_methylation_test_new)
        data_methylation_test_new.columns=sample_test


        data_CNV_train_new=pca_CNV.transform(np.array(data_CNV_train_scale).T).T
        data_CNV_train_new=pd.DataFrame(data_CNV_train_new)
        data_CNV_train_new.columns=sample_train
        data_CNV_validation_new = pca_CNV.transform(np.array(data_CNV_validation_scale).T).T
        data_CNV_validation_new=pd.DataFrame(data_CNV_validation_new)
        data_CNV_validation_new.columns=sample_validation
        data_CNV_test_new = pca_CNV.transform(np.array(data_CNV_test_scale).T).T
        data_CNV_test_new=pd.DataFrame(data_CNV_test_new)
        data_CNV_test_new.columns=sample_test


        data_count_train_all=pd.concat([data_count_train_all,data_count_train_new])
        data_count_validation_all=pd.concat([data_count_validation_all,data_count_validation_new])
        data_count_test_all=pd.concat([data_count_test_all,data_count_test_new])
        data_methylation_train_all=pd.concat([data_methylation_train_all,data_methylation_train_new])
        data_methylation_validation_all=pd.concat([data_methylation_validation_all,data_methylation_validation_new])
        data_methylation_test_all=pd.concat([data_methylation_test_all,data_methylation_test_new])
        data_CNV_train_all=pd.concat([data_CNV_train_all,data_CNV_train_new])
        data_CNV_validation_all=pd.concat([data_CNV_validation_all,data_CNV_validation_new])
        data_CNV_test_all=pd.concat([data_CNV_test_all,data_CNV_test_new])


    data_count_train_all.to_csv(save_path+str(dataset)+'/data_count_train_all.txt',sep='\t')
    data_count_validation_all.to_csv(save_path+str(dataset)+'/data_count_validation_all.txt',sep='\t')
    data_count_test_all.to_csv(save_path+str(dataset)+'/data_count_test_all.txt',sep='\t')

    data_methylation_train_all.to_csv(save_path+str(dataset)+'/data_methylation_train_all.txt',sep='\t')
    data_methylation_validation_all.to_csv(save_path+str(dataset)+'/data_methylation_validation_all.txt',sep='\t')
    data_methylation_test_all.to_csv(save_path+str(dataset)+'/data_methylation_test_all.txt',sep='\t')

    data_CNV_train_all.to_csv(save_path+str(dataset)+'/data_CNV_train_all.txt',sep='\t')
    data_CNV_validation_all.to_csv(save_path+str(dataset)+'/data_CNV_validation_all.txt',sep='\t')
    data_CNV_test_all.to_csv(save_path+str(dataset)+'/data_CNV_test_all.txt',sep='\t')



def main(args):
    get_pac(args.dataset, args.label_path, args.data_path,args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)
