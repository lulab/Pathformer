import pandas as pd
import numpy as np
import argparse

def get_sample(smaple,type='RNA'):
    smaple=pd.DataFrame(smaple)
    smaple.columns=['ID']
    smaple['sample_id']=smaple['ID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
    smaple_over=list(set(smaple['sample_id']))
    smaple_1=pd.DataFrame(columns=['ID','sample_id'])
    smaple_1['sample_id']=smaple_over
    for s in smaple_over:
        id_data=smaple.loc[smaple['sample_id']==s,['ID']]
        id_data['Sample']=id_data['ID'].map(lambda x:x.split('-')[3][:2])
        id_data['Vial'] = id_data['ID'].map(lambda x: x.split('-')[3][2:])
        id_data['Portion'] = id_data['ID'].map(lambda x: x.split('-')[4][:2])
        id_data['Analyte'] = id_data['ID'].map(lambda x: x.split('-')[4][2:])
        id_data['Plate'] = id_data['ID'].map(lambda x: x.split('-')[5])
        id_data['Center'] = id_data['ID'].map(lambda x: x.split('-')[6])
        id_data=id_data.loc[id_data.Sample.isin(['01','02','03','04','05','06','07','08','09'])]
        id_data = id_data.loc[id_data['Vial']!='B']
        if type=='RNA':
            id_data = id_data.loc[id_data.Analyte.isin(['H','R','T'])]
        else:
            id_data = id_data.loc[id_data.Analyte.isin(['D','G','W','X'])]
        if len(id_data)==0:
            smaple_1.loc[smaple_1['sample_id']==s,'ID']=np.nan
            continue
        elif len(id_data)==1:
            smaple_1.loc[smaple_1['sample_id']==s,'ID']=list(id_data['ID'])
            continue
        else:
            if type=='RNA':
                Analyte_list = list(set(id_data['Analyte']))
                Analyte_list.sort()
                id_data= id_data.loc[id_data['Analyte']==Analyte_list[0]]
            else:
                if 'D' in list(set(id_data['Analyte'])):
                    id_data = id_data.loc[id_data['Analyte'] == 'D']
            if len(id_data)==1:
                smaple_1.loc[smaple_1['sample_id']==s,'ID']=list(id_data['ID'])
                continue
            else:
                Vial_list = list(set(id_data['Vial']))
                Vial_list.sort()
                id_data= id_data.loc[id_data['Vial']==Vial_list[0]]
                if len(id_data) == 1:
                    smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                    continue
                else:
                    Sample_list = list(set(id_data['Sample']))
                    Sample_list.sort()
                    id_data = id_data.loc[id_data['Sample'] == Sample_list[0]]
                    if len(id_data) == 1:
                        smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                        continue
                    else:
                       Portion_list=list(set(id_data['Portion'].astype('int')))
                       Portion_list.sort()
                       Portion=str(Portion_list[-1])
                       if len(Portion)==1:Portion='0'+Portion
                       id_data = id_data.loc[id_data['Portion'] ==Portion]
                       if len(id_data) == 1:
                           smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                           continue
                       else:
                           Plate_list = list(set(id_data['Plate']))
                           Plate_list.sort()
                           id_data = id_data.loc[id_data['Plate'] == str(Plate_list[-1])]
                           if len(id_data) == 1:
                               smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                               continue
                           else:
                               Center_list = list(set(id_data['Center'].astype('int')))
                               Center_list.sort()
                               Center = str(Center_list[0])
                               if len(Center) == 1: Center = '0' + Center
                               id_data = id_data.loc[id_data['Center'] == Center]
                               smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])

    smaple_1=smaple_1.loc[pd.notnull(smaple_1['ID'])]
    return smaple_1



def data_filter(raw_data_path,embedding_data_path,save_path,cancer):
    print('miRNA')
    data_miRNA=pd.read_csv(raw_data_path+cancer+'.miRNA.csv',sep=',',nrows=1)
    sample_miRNA=data_miRNA.columns[2:]
    sample_miRNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
    sample_miRNA_data['sample_old']=sample_miRNA
    sample_miRNA_data['type']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_TCGA')[0])
    sample_miRNA_data=sample_miRNA_data.loc[sample_miRNA_data['type']=='read_count',:]
    sample_miRNA_data['sample_new']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_')[-1])
    sample_miRNA_data['sample_id']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_')[-1].split('-')[0]+'-'+x.split('_')[-1].split('-')[1]+'-'+x.split('_')[-1].split('-')[2])
    sample_miRNA_filter_data=get_sample(list(sample_miRNA_data['sample_new']))
    sample_miRNA_data=sample_miRNA_data.loc[sample_miRNA_data.sample_new.isin(list(sample_miRNA_filter_data['ID'])),:]

    print('mRNA')
    data_mRNA=pd.read_csv(raw_data_path+cancer+'.mRNA.csv',sep=',',nrows=1)
    sample_mRNA=data_mRNA.columns[1:]
    sample_mRNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
    sample_mRNA_data['sample_old']=sample_mRNA
    sample_mRNA_data['sample_id']=sample_mRNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
    sample_mRNA_filter_data=get_sample(list(sample_mRNA_data['sample_old']))
    sample_mRNA_data=sample_mRNA_data.loc[sample_mRNA_data.sample_old.isin(list(sample_mRNA_filter_data['ID'])),:]

    print('DNAmethy')
    data_DNA=pd.read_csv(embedding_data_path+'methylation_count.txt',sep='\t',nrows=1)
    sample_DNA=data_DNA.columns[1:]
    sample_DNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
    sample_DNA_data['sample_old']=sample_DNA
    sample_DNA_data['sample_id']=sample_DNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
    sample_DNA_filter_data=get_sample(list(sample_DNA_data['sample_old']),type='DNA')
    sample_DNA_data=sample_DNA_data.loc[sample_DNA_data.sample_old.isin(list(sample_DNA_filter_data['ID'])),:]

    print('CNV')
    data_CNV=pd.read_csv(raw_data_path+cancer+'.CNV_masked_seg.csv',sep=',')
    sample_CNV=list(set(data_CNV['Sample']))
    sample_CNV_data=pd.DataFrame(columns=['sample_old','sample_id'])
    sample_CNV_data['sample_old']=sample_CNV
    sample_CNV_data['sample_id']=sample_CNV_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
    sample_CNV_filter_data=get_sample(list(sample_CNV_data['sample_old']),type='DNA')
    sample_CNV_data=sample_CNV_data.loc[sample_CNV_data.sample_old.isin(list(sample_CNV_filter_data['ID'])),:]

    print('CNV_gene')
    data_CNV_gene=pd.read_csv(embedding_data_path+'all_data_by_genes.txt',sep='\t',nrows=1)
    sample_CNV_gene=data_CNV_gene.columns[3:]
    sample_CNV_gene_data=pd.DataFrame(columns=['sample_old','sample_id'])
    sample_CNV_gene_data['sample_old']=sample_CNV_gene
    sample_CNV_gene_data['sample_id']=sample_CNV_gene_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
    sample_CNV_gene_filter_data=get_sample(list(sample_CNV_gene_data['sample_old']),type='DNA')
    sample_CNV_gene_data=sample_CNV_gene_data.loc[sample_CNV_gene_data.sample_old.isin(list(sample_CNV_gene_filter_data['ID'])),:]


    print(len(set(sample_mRNA_data['sample_id'])))
    print(len(set(sample_miRNA_data['sample_id'])))
    print(len(set(sample_DNA_data['sample_id'])))
    print(len(set(sample_CNV_data['sample_id'])))
    print(len(set(sample_CNV_gene_data['sample_id'])))

    sample_id_over_2=list(set(sample_mRNA_data['sample_id'])&set(sample_DNA_data['sample_id'])&set(sample_CNV_data['sample_id'])&set(sample_CNV_gene_data['sample_id']))
    print(len(sample_id_over_2))

    sample_mRNA_data.loc[sample_mRNA_data.sample_id.isin(sample_id_over_2)].to_csv(save_path+'/sample_mRNA_data_2.txt',sep='\t',index=False)
    sample_miRNA_data.loc[sample_miRNA_data.sample_id.isin(sample_id_over_2)].to_csv(save_path+'/sample_miRNA_data_2.txt',sep='\t',index=False)
    sample_DNA_data.loc[sample_DNA_data.sample_id.isin(sample_id_over_2)].to_csv(save_path+'/sample_DNA_data_2.txt',sep='\t',index=False)
    sample_CNV_data.loc[sample_CNV_data.sample_id.isin(sample_id_over_2)].to_csv(save_path+'/sample_CNV_data_2.txt',sep='\t',index=False)
    sample_CNV_gene_data.loc[sample_CNV_gene_data.sample_id.isin(sample_id_over_2)].to_csv(save_path+'/sample_CNV_gene_data_2.txt',sep='\t',index=False)

def main(args):
    data_filter(args.raw_data_path, args.embedding_data_path, args.save_path, args.cancer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine learning module')
    parser.add_argument('--raw_data_path', type=str, required=True,
                        help='raw_data_path', dest='raw_data_path')
    parser.add_argument('--embedding_data_path', type=str, required=True,
                        help='embedding_data_path', dest='embedding_data_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cancer', type=str, required=True,
                        help='cancer', dest='cancer')
    args = parser.parse_args()
    main(args)