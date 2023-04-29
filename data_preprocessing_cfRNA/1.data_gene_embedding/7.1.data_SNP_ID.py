import pandas as pd
import numpy as np
import re

def get_id(x,gtf_data_new):
    # print(x)
    chr=x.split('_')[0]
    site=int(x.split('_')[-3].split('.')[0])
    gtf=gtf_data_new.loc[(gtf_data_new['chr']==chr)&(gtf_data_new['strat']<=site)&(gtf_data_new['end']>=site),:]
    if len(set(gtf['strand']))>1:
        gtf=gtf.loc[gtf['strand']=='+',:]
    if len(gtf)==0:
        gene_id='NA'
        gene_name='NA'
        # print('erro')
    else:
        gene_id=list(gtf['gene_id'])[0]
        gene_name=list(gtf['gene_name'])[0]
    #id=x+'_'+gene_id+'_'+gene_name
    return gene_id

def get_ID_all(input_path,reference_path,save_path):
    data=pd.read_csv(input_path+'/SNP_rowdata.txt',sep='\t')
    data_ID=pd.DataFrame(data['ID'])
    del data

    gtf_data=pd.read_csv(reference_path+'/gencode.v27.annotation.gtf',sep='\t',skiprows = lambda x: x in [0,1,2,3,4],header=None)
    gtf_data=gtf_data.loc[gtf_data.iloc[:,2]=='gene',:]
    gtf_data_new = pd.DataFrame(columns={'gene_id','gene_name','chr','strat','end','strand'})
    gtf_data_new['gene_id'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_id ".*?"',x)[0].split('"')[1])
    gtf_data_new['gene_name'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_name ".*?"',x)[0].split('"')[1] if 'gene_name' in x else np.nan)
    gtf_data_new['chr'] = gtf_data.iloc[:,0]
    gtf_data_new['strat']=gtf_data.iloc[:,3].astype('int')
    gtf_data_new['end']=gtf_data.iloc[:,4].astype('int')
    gtf_data_new['strand']=gtf_data.iloc[:,6]
    gtf_data_new = gtf_data_new.drop_duplicates()
    gtf_data_new.index = range(len(gtf_data_new))

    if len(data)<=10000:
        data['ID_new']=data['ID'].map(lambda x: get_id(x, gtf_data_new))
    else:
        for j in [i*10000 for i in range(0,int(len(data_ID)/10000)+1)]:
            print(j)
            data_ID.loc[j:(j+10000),'ID_new'] = data_ID.loc[j:(j+10000),'ID'].map(lambda x: get_id(x, gtf_data_new))
        data_ID.loc[int(len(data_ID)/10000)*10000:,'ID_new'] = data_ID.loc[int(len(data_ID)/10000)*10000:,'ID'].map(lambda x: get_id(x, gtf_data_new))

    data_ID=data_ID.rename(columns={'ID':'feature','ID_new':'gene_id'})
    data_ID.to_csv(save_path+'/SNP_ID.txt',sep='\t',index=False)

def main(args):
    get_ID_all(args.input_path,args.reference_path, args.save_path)

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