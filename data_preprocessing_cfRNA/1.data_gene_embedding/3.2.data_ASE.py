import pandas as pd
import numpy as np
from numba import jit
import scipy.stats

# @jit(nopython=False)
def get_entropy(x):
    entropy_list=[]
    for i in sample:
        arr=np.array(x[i])
        if len(set(arr))==1:
            entropy=0
        else:
            entropy=scipy.stats.entropy(arr[arr>0])
        entropy_list.append(entropy)
    data=pd.Series(entropy_list)
    data.index=sample
    return data


def get_feature(input_path,reference_path,save_path):
    data=pd.read_csv(input_path+'/ASE_rawdata.txt',sep='\t')
    data=data.fillna(0)
    sample=list(data.columns[1:])
    ID=pd.read_csv(save_path+'/ASE_ID.txt',sep='\t')
    ID=ID.rename(columns={'feature':'ID'})
    data=pd.merge(data,ID[['ID','gene_id']],on='ID',how='left')
    data[data==0]=np.nan

    #max
    data_max=data[['gene_id']+sample].groupby('gene_id').max()
    data_max=data_max.fillna(0)
    data_max=data_max.reset_index()
    data_max.to_csv(save_path+'/ASE_max.txt',sep='\t',index=False)

    #min
    data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
    data_min=data_min.fillna(0)
    # data_min=data_min.reset_index()
    data_min.to_csv(save_path+'/ASE_min.txt',sep='\t',index=False)

    #mean
    data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
    data_mean=data_mean.fillna(0)
    data_mean=data_mean.reset_index()
    data_mean.to_csv(save_path+'/ASE_mean.txt',sep='\t',index=False)

    #count
    data=data.fillna(0)
    data_=data[sample]
    data_[data_>0]=1
    data_['gene_id']=data['gene_id']
    data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
    data_count=data_count.fillna(0)
    data_count=data_count.reset_index()
    data_count.to_csv(save_path+'/ASE_count.txt',sep='\t',index=False)

    #entropy
    data_entropy=data.loc[:,['gene_id']+sample].groupby('gene_id').apply(get_entropy)
    data_entropy=data_entropy.fillna(0)
    data_entropy=data_entropy.reset_index()
    data_entropy.to_csv(save_path+'/ASE_entropy.txt',sep='\t',index=False)


    #all
    depth_all=pd.read_csv(input_path+'/ASE_depth_sum.txt',sep='\t')
    depth_all.columns=[i.split('.')[0] for i in list(depth_all.columns)]
    depth_all=pd.merge(depth_all,ID[['ID','gene_id']],on='ID',how='left')
    depth_alt=pd.read_csv(input_path+'/ASE_alt_depth.txt',sep='\t')
    depth_alt.columns=[i.split('.')[0] for i in list(depth_alt.columns)]
    depth_alt=pd.merge(depth_alt,ID[['ID','gene_id']],on='ID',how='left')

    depth_all=depth_all.fillna(0)
    depth_alt=depth_alt.fillna(0)
    depth_all=depth_all[['gene_id']+sample].groupby('gene_id').sum()
    depth_alt=depth_alt[['gene_id']+sample].groupby('gene_id').sum()

    ratio=depth_alt/depth_all
    ratio=ratio.fillna(0)
    ratio=ratio.reset_index()

    ratio.to_csv(save_path+'/ASE_mean_all.txt',sep='\t',index=False)

    #window
    depth_all=pd.read_csv(input_path+'/ASE_depth_sum.txt',sep='\t')
    depth_all.columns=[i.split('.')[0] for i in list(depth_all.columns)]
    depth_all=pd.merge(depth_all,ID[['ID','gene_id']],on='ID',how='left')
    depth_alt=pd.read_csv(input_path+'/ASE_alt_depth.txt',sep='\t')
    depth_alt.columns=[i.split('.')[0] for i in list(depth_alt.columns)]
    depth_alt=pd.merge(depth_alt,ID[['ID','gene_id']],on='ID',how='left')
    depth_all=depth_all.fillna(0)
    depth_alt=depth_alt.fillna(0)
    gtf=pd.read_csv(reference_path+'/gene_windows.txt',sep='\t')

    def get_location(x):
        gtf_ = gtf.loc[gtf['gene_id'] == x['gene_id']]
        if len(gtf_) == 0:
            window = np.nan
        else:
            try:
                locat = float(x['ID'].split('|')[1])
                strat = int(list(gtf_['strat'])[0])
                end = int(list(gtf_['end'])[0])
                cut_1 = int(list(gtf_['cut_1'])[0])
                cut_2 = int(list(gtf_['cut_2'])[0])
                if (locat >= strat) & (locat <= cut_1):
                    window = 1
                elif (locat > cut_1) & (locat <= cut_2):
                    window = 2
                elif (locat > cut_2) & (locat <= end):
                    window = 3
                else:
                    window = np.nan
            except ValueError:
                window = np.nan
        return window

    depth_all['location']=depth_all.loc[:,['ID','gene_id']].apply(get_location,axis=1)
    depth_alt=pd.merge(depth_alt,depth_all[['ID','location']],on='ID',how='left')

    depth_all_1=depth_all.loc[depth_all['location']==1,:]
    depth_all_2=depth_all.loc[depth_all['location']==2,:]
    depth_all_3=depth_all.loc[depth_all['location']==3,:]
    depth_alt_1=depth_alt.loc[depth_alt['location']==1,:]
    depth_alt_2=depth_alt.loc[depth_alt['location']==2,:]
    depth_alt_3=depth_alt.loc[depth_alt['location']==3,:]

    depth_all_1=depth_all_1[['gene_id']+sample].groupby('gene_id').sum()
    depth_alt_1=depth_alt_1[['gene_id']+sample].groupby('gene_id').sum()
    ratio_1=depth_alt_1/depth_all_1
    ratio_1=ratio_1.fillna(0)
    ratio_1=ratio_1.reset_index()
    ratio_1.to_csv(save_path+'/ASE_mean_window_1.txt',sep='\t',index=False)

    depth_all_2=depth_all_2[['gene_id']+sample].groupby('gene_id').sum()
    depth_alt_2=depth_alt_2[['gene_id']+sample].groupby('gene_id').sum()
    ratio_2=depth_alt_2/depth_all_2
    ratio_2=ratio_2.fillna(0)
    ratio_2=ratio_2.reset_index()
    ratio_2.to_csv(save_path+'/ASE_mean_window_2.txt',sep='\t',index=False)

    depth_all_3=depth_all_3[['gene_id']+sample].groupby('gene_id').sum()
    depth_alt_3=depth_alt_3[['gene_id']+sample].groupby('gene_id').sum()
    ratio_3=depth_alt_3/depth_all_3
    ratio_3=ratio_3.fillna(0)
    ratio_3=ratio_3.reset_index()
    ratio_3.to_csv(save_path+'/ASE_mean_window_3.txt',sep='\t',index=False)

def main(args):
    get_feature(args.input_path,args.reference_path, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--input_path', type=str, required=True,
        help='input_path',dest='input_path')
    parser.add_argument('--reference_path', type=str, required=True,
        help='reference_path',dest='reference_path')
    parser.add_argument('--save_path', type=str, required=True,
        help='save_path',dest='save_path')
    args = parser.parse_args()
    main(args)