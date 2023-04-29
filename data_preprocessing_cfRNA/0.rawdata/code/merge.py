import pandas as pd
import numpy as np

###AS
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/AS/GSE133684_inclevel.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/AS/AS_merge_all.txt',sep='\t')

data_1=data_1.reset_index().rename(columns={'index':'feature'})
data_2=data_2.rename(columns={'Unnamed: 0':'feature'})

data_all=pd.merge(data_1,data_2,on='feature',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/AS_rawdata.txt',sep='\t',index=False)

###ASE
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/ASE/ALL_alt_depth.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/ASE/alt_depth_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/ASE_alt_depth.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/ASE/ALL_depth_sum.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/ASE/depth_sum_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/ASE_depth_sum.txt',sep='\t',index=False)


data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/ASE/ASE_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/ASE/ASE_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/ASE_rawdata.txt',sep='\t',index=False)

###chimeric
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/chimeric/chimeric_data.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/chimeric/GSE133684_chimeric.txt',sep='\t')

data_2=data_2.rename(columns={'Chimeric':'ID'})
data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/chimericRNA_rawdata.txt',sep='\t',index=False)

###count
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/count/featurecounts_noMTRNA.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/count/count-matrix_noMTRNA.txt',sep='\t')

data_2=data_2.reset_index().rename(columns={'index':'feature'})

data_all=pd.merge(data_1,data_2,on='feature',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/expression_rawdata.txt',sep='\t',index=False)

###editing
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/editing/editing_all_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/editing/editing_all.csv',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/editing_rawdata.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/editing/Editing_totalCount_new_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/editing/Editing_totalCount_new.csv',sep='\t')

data_all=pd.merge(data_2,data_1,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/editing_totalcount_rawdata.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/editing/Editing_altCount_new_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/editing/Editing_altCount_new.csv',sep='\t')

data_all=pd.merge(data_2,data_1,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/editing_altcount_rawdata.txt',sep='\t',index=False)

###promoter
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/promoter/TPM-by-promoter_exoRBase.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/promoter/TPM-by-promoter_GSE133684.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='promoter_id',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/promoter_rawdata.txt',sep='\t',index=False)

###SNP
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/SNP/depth_sum_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/SNP/ALL_depth_sum.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/SNP_depth_sum.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/SNP/alt_depth_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/SNP/ALL_alt_depth.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/SNP_alt_depth.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/exoRbase/SNP/SNP_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/gut/SNP/SNP_PDAC.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/EV/SNP_rawdata.txt',sep='\t',index=False)
