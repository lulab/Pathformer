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


def get_attention_weight(modal_all_path,save_path):
    data_label=np.load(file=save_path+'/data_label.npy')
    data_label=pd.DataFrame(data_label)
    data_label.columns=['y']
    label_list=list(set(data_label['y']))
    attn_col=h5py.File(save_path+'/attn_out_col_all.h5', 'r')
    modal_name_all=list(pd.read_csv(modal_all_path,header=None)[0])

    data=pd.DataFrame(columns={'modal_type'})
    data['modal_type']=modal_name_all

    label_index=list(data_label.index+1)
    attn_col_all=np.zeros([len(label_index),len(modal_name_all)])
    for s in range(len(label_index)):
        attn_col_=attn_col[str(label_index[s])][:][0]
        attn_col_=attn_col_.mean(axis=0)
        attn_col_all[s,:]=attn_col_
    attn_col_important= attn_col_all.mean(axis=0)
    data['attention_weight']=attn_col_important
    data.to_csv(save_path+'important_omics.txt',sep='\t',index=False)

    plt.figure(figsize=(7,8)) #调节画布的大小
    labels = list(data['modal_type']) #定义各个扇形的面积/标签
    sizes = list(data['attention_weight']) #各个值，影响各个扇形的面积
    patches, texts = plt.pie(sizes,shadow = False,startangle =90)
    plt.axis('equal')
    plt.legend(patches, labels, loc=9,bbox_to_anchor=(0.5,0.15),ncol=2)
    # plt.show()
    plt.savefig(save_path+'/important_omics_pie.pdf')
    plt.close()


def main(args):
    get_attention_weight(args.modal_all_path,args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--modal_all_path', type=str, default='None',
                        help='modal_all_path', dest='modal_all_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)


