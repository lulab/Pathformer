import pandas as pd
import numpy as np
import re
import argparse

def get_data(reference_path,rawdata_path,save_path,cancer):
    data_CNV = pd.read_csv(rawdata_path+'/'+cancer+'.CNV_masked_seg.csv', sep=',')
    data_CNV['ID'] = 'chr' + data_CNV['Chromosome'].astype(str) + '_' + data_CNV['Start'].astype(str) + '_' + data_CNV['End'].astype(str)
    ID = data_CNV[['ID']].drop_duplicates()

    gtf_data = pd.read_csv(reference_path+'/Homo_sapiens.GRCh38.91.chr.gtf', sep='\t',skiprows=lambda x: x in [0, 1, 2, 3, 4], header=None)
    gtf_data = gtf_data.loc[gtf_data.iloc[:, 2] == 'gene', :]
    gtf_data_new = pd.DataFrame(columns=['gene_id', 'gene_name', 'chr', 'strat', 'end', 'strand'])
    gtf_data_new['gene_id'] = gtf_data.iloc[:, 8].apply(lambda x: re.findall('gene_id ".*?"', x)[0].split('"')[1])
    gtf_data_new['gene_name'] = gtf_data.iloc[:, 8].apply(lambda x: re.findall('gene_name ".*?"', x)[0].split('"')[1] if 'gene_name' in x else np.nan)
    gtf_data_new['chr'] = gtf_data.iloc[:, 0].astype('str')
    gtf_data_new['strat'] = gtf_data.iloc[:, 3].astype('int')
    gtf_data_new['end'] = gtf_data.iloc[:, 4].astype('int')
    gtf_data_new['strand'] = gtf_data.iloc[:, 6]
    gtf_data_new = gtf_data_new.drop_duplicates()
    gtf_data_new.index = range(len(gtf_data_new))


    def get_id(x, gtf_data_new):
        # print(x)
        chr = x.split('_')[0].split('chr')[1]
        site_strat = int(x.split('_')[1])
        site_end = int(x.split('_')[2])

        gtf = gtf_data_new.loc[
              ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] <= site_strat) & (gtf_data_new['end'] >= site_strat)) |
              ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] <= site_end) & (gtf_data_new['end'] >= site_end)) |
              ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] >= site_strat) & (gtf_data_new['end'] <= site_end)), :]
        gtf = gtf.drop_duplicates('gene_id')
        if len(gtf) == 0:
            gene_id = 'NA'
            # print('erro')
        else:
            gene_id = ';'.join(list(gtf['gene_id']))
        # id=x+'_'+gene_id+'_'+gene_name
        return gene_id


    # data['ID_new']=data['ID'].map(lambda x: get_id(x, gtf_data_new))
    for j in [i * 10000 for i in range(0, int(len(ID) / 10000) + 1)]:
        print(j)
        ID.loc[j:(j + 10000), 'ID_new'] = ID.loc[j:(j + 10000), 'ID'].map(lambda x: get_id(x, gtf_data_new))
    ID.loc[int(len(ID) / 10000) * 10000:, 'ID_new'] = ID.loc[int(len(ID) / 10000) * 10000:, 'ID'].map(
        lambda x: get_id(x, gtf_data_new))
    ID['gene_id'] = ID['ID_new'].str.split(';')
    data_ID_new = ID.explode('gene_id')
    data_ID_new = data_ID_new[['ID', 'gene_id']]
    data_ID_new = data_ID_new.drop_duplicates()
    data_ID_new.to_csv(save_path+'/CNV_id.txt', sep='\t', index=False)

def main(args):
    get_data(args.reference_path,args.rawdata_path,args.save_path,args.cancer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='reference_path', dest='reference_path')
    parser.add_argument('--rawdata_path', type=str, required=True,
                        help='rawdata_path', dest='rawdata_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    parser.add_argument('--cancer', type=str, required=True,
                        help='cancer', dest='cancer')
    args = parser.parse_args()
    main(args)