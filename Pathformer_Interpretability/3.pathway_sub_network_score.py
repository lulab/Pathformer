import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import argparse


def get_sub_network_score(pathway_name_path,pathway_crosstalk_network,save_path,cut_off):
    pathway_data = pd.read_csv(pathway_name_path, sep='\t')
    pathway_name = list(pathway_data['name'])
    pathway_network = np.load(file=pathway_crosstalk_network)
    pathway_network[np.isnan(pathway_network)] = 0
    pathway_network=pd.DataFrame(pathway_network)
    pathway_network.columns=pathway_name
    pathway_network.index=pathway_name

    data=pd.read_csv(save_path+'/shap_value_pathway.txt',sep='\t')
    pathway_100=list(data.loc[data['length']<=100,'pathway'])
    data_label=np.load(file=save_path+'/data_label.npy')
    data_label=pd.DataFrame(data_label)
    data_label.columns=['y']
    label_list=list(set(data_label['y']))
    label_index = list(data_label.index + 1)

    net_data=h5py.File(save_path+'/net_all.h5', 'r')
    net_update_data_=np.zeros([len(pathway_data),len(pathway_data)])
    for s in range(len(label_index)):
        if s%100==0:
            print(s)
        net_data_=net_data[str(label_index[s])][:]
        net_update_data_=net_update_data_+net_data_
    net_update_data_=net_update_data_/len(label_index)
    for i in range(len(net_update_data_)):
        net_update_data_[i,i]=np.nan
    net_update_data=(net_update_data_-np.nanmin(net_update_data_))/(np.nanmax(net_update_data_)-np.nanmin(net_update_data_))
    for i in range(len(net_update_data)):
        net_update_data[i,i]=1
    np.save(file=save_path+'pathway_crosstalk_network_update.npy', arr=np.array(net_update_data))
    net_data.close()

    net_update_data=pd.DataFrame(net_update_data)
    net_update_data.columns=pathway_name
    net_update_data.index = pathway_name
    net_update_data_select=net_update_data.loc[pathway_100,pathway_100]

    ###important module####
    cut = np.quantile(np.array(net_update_data_select), cut_off)
    data_module_all=pd.DataFrame(columns={'center_pathway','sub_network_score','sub_network_number','sub_network_pathways'})
    data_module_all['center_pathway']=pathway_100
    for i in range(len(net_update_data_select)):
        net_all_=net_update_data_select.loc[:,net_update_data_select.iloc[i,:]>cut]
        net_all_select = net_all_.loc[(net_all_ > cut).sum(axis=1)>0, :]
        pathway_select=list(set(net_all_select.index)|set(net_all_select.columns))
        data_module_all.loc[i,'sub_network_score']=data.loc[data.pathway.isin(pathway_select),'shap_pathway'].mean()
        data_module_all.loc[i,'sub_network_number']=len(pathway_select)
        data_module_all.loc[i, 'sub_network_pathways'] = ','.join(pathway_select)
        gene_select = ','.join(list(data.loc[data.pathway.isin(pathway_select),'gene']))
        data_module_all.loc[i, 'gene_number'] =len(set(gene_select.split(',')))

    data_module_all=data_module_all.loc[(data_module_all['sub_network_number']>1),:]
    data_module_all=data_module_all.sort_values('sub_network_score',ascending=False)
    pathway_select_center=list(data_module_all['center_pathway'])[0]
    pathway_select_hub_modul=list(data_module_all['sub_network_pathways'])[0].split(',')


    data_hub_modul_weigth=pd.DataFrame(columns=['Source','Target','pathway_net_old','pathway_net_update'])
    i=0
    for p1 in pathway_select_hub_modul:
        for p2 in pathway_select_hub_modul:
            data_hub_modul_weigth.loc[i,'Source']=p1
            data_hub_modul_weigth.loc[i, 'Target'] = p2
            data_hub_modul_weigth.loc[i, 'pathway_net_old'] = abs(pathway_network.loc[p1, p2])
            data_hub_modul_weigth.loc[i, 'pathway_net_update'] = abs(net_update_data.loc[p1,p2])
            i = i + 1
    data_hub_modul_weigth['pathway_net_link_update']=data_hub_modul_weigth['pathway_net_update'].map(lambda x:1 if x>cut else 0)
    data_hub_modul_weigth_new=data_hub_modul_weigth.loc[data_hub_modul_weigth['pathway_net_link_update']>0,:]
    data_hub_modul_weigth_new.index=range(len(data_hub_modul_weigth_new))

    for i in range(len(data_hub_modul_weigth_new)):
        if  (data_hub_modul_weigth_new.loc[i,'pathway_net_link_update']==1)&(data_hub_modul_weigth_new.loc[i,'pathway_net_old']==0):
            data_hub_modul_weigth_new.loc[i, 'link_new'] = 1
        else:
            data_hub_modul_weigth_new.loc[i, 'link_new'] = 0
    data_hub_modul_weigth_new=data_hub_modul_weigth_new[['Source','Target','pathway_net_old','pathway_net_update','pathway_net_link_update','link_new']]


    data_module_all[['center_pathway','sub_network_score','sub_network_number','sub_network_pathways','gene_number']].to_csv(save_path+'/pathway_sub_network_score_all.txt',sep='\t',index=False)
    pd.DataFrame(pathway_select_hub_modul).to_csv(save_path+'/pathway_network_hub_modul_pathway.txt',sep='\t',index=False,header=False)
    data_hub_modul_weigth_new.to_csv(save_path+'/pathway_network_hub_modul_weight.txt',sep=',',index=False)


def main(args):
    get_sub_network_score(args.pathway_name_path, args.pathway_crosstalk_network, args.save_path, args.cut_off)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--pathway_name_path', type=str, required=True,
                        help='pathway_name_path', dest='pathway_name_path')
    parser.add_argument('--pathway_crosstalk_network', type=str, required=True,
                        help='pathway_crosstalk_network', dest='pathway_crosstalk_network')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cut_off', type=float, required=True,
                        help='cut_off', dest='cut_off')
    args = parser.parse_args()
    main(args)
