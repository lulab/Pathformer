import pandas as pd
import numpy as np

# ###AS
# data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/AS/AS_all.txt',sep='\t')
# data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/AS/AS_all.txt',sep='\t')
#
# data_all=pd.merge(data_1,data_2,on='feature',how='outer')
# data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/AS_rawdata.txt',sep='\t',index=False)
#
# ###ASE
# data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/ASE/alt_depth_all.txt',sep='\t')
# data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/ASE/alt_depth_all.txt',sep='\t')
#
# data_all=pd.merge(data_1,data_2,on='ID',how='outer')
# data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/ASE_alt_depth.txt',sep='\t',index=False)
#
# data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/ASE/depth_sum_all.txt',sep='\t')
# data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/ASE/depth_sum_all.txt',sep='\t')
#
# data_all=pd.merge(data_1,data_2,on='ID',how='outer')
# data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/ASE_depth_sum.txt',sep='\t',index=False)


data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/ASE/ASE_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/ASE/ASE_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/ASE_rawdata.txt',sep='\t',index=False)
print('ASE')
###chimeric
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/chimeric/GSE68086_chimeric.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/chimeric/GSE89843_chimeric.txt',sep='\t')
data_1=data_1.rename(columns={'Chimeric':'ID'})
data_2=data_2.rename(columns={'Chimeric':'ID'})
data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/chimericRNA_rawdata.txt',sep='\t',index=False)
print('chimeric')
###count
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/count/GSE68086_count_matrix_noMTRNA.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/count/GSE89843_count_matrix_noMTRNA.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='feature',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/expression_rawdata.txt',sep='\t',index=False)
print('count')
###editing
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/editing/editing_all_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/editing/editing_all.csv',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/editing_rawdata.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/editing/Editing_totalCount_new_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/editing/Editing_totalCount_new_50.csv',sep='\t')

data_all=pd.merge(data_2,data_1,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/editing_totalcount_rawdata.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/editing/Editing_altCount_new_50.csv',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/editing/Editing_altCount_new_50.csv',sep='\t')

data_all=pd.merge(data_2,data_1,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/editing_altcount_rawdata.txt',sep='\t',index=False)
print('editing')
###promoter
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/promoter/TPM-by-promoter_GSE68086.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/promoter/TPM-by-promoter_GSE89843.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='promoter_id',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/promoter_rawdata.txt',sep='\t',index=False)
print('promoter')
###SNP
data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/SNP/depth_sum_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/SNP/depth_sum_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/SNP_depth_sum.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/SNP/alt_depth_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/SNP/alt_depth_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/SNP_alt_depth.txt',sep='\t',index=False)

data_1=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2015/SNP/SNP_all.txt',sep='\t')
data_2=pd.read_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/cfRNA/RNA_rawdata/TEP_2017/SNP/SNP_all.txt',sep='\t')

data_all=pd.merge(data_1,data_2,on='ID',how='outer')
data_all.to_csv('/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_preprocessing_cfRNA/0.rawdata/platelet/SNP_rawdata.txt',sep='\t',index=False)
print('SNP')