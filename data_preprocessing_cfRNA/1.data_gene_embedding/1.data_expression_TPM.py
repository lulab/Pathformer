import numpy as np
import pandas as pd

def get_feature(input_path,reference_path,save_path):
    df = pd.read_csv(input_path+'/expression_rawdata.txt',sep="\t")
    sample=list(df.columns[1:])
    df['ID']=df['feature'].map(lambda x:x.split('|')[0].split('_PAR_Y')[0])
    df_new=df[['ID']+sample].groupby('ID').sum()
    length_data=pd.read_csv(reference_path+'/gene_length_cfRNA.txt',sep='\t')


    print("Done .")
    print("Calculate TPM ...")
    gene = list(df_new.index)
    length_data =length_data.set_index('gene')
    length=length_data.loc[gene,:]
    lengthScaledDf = pd.DataFrame((df_new.values/length.values.reshape((-1,1))),index=df_new.index,columns=df_new.columns)
    data_1 = (1000000*lengthScaledDf.div(lengthScaledDf.sum(axis=0))).round(4)
    data_1=data_1.reset_index()

    data_1.to_csv(save_path+'/ALL_data_TPM.txt',sep='\t',index=False)

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
