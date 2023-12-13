# Liquid biopsy data

We used three types of body fluid datasets: the plasma dataset, the extracellular vesicle (EV) dataset, and the platelet dataset. Liquid biopsies data preprocessing process can be seen in the folder: */data_preprocessing_cfRNA*.

The following shows the directory tree of data folder. Due to storage space limitations on Github, data can be  downloaded from these links (链接: https://pan.baidu.com/s/15K1tyV1Skrb7KEcDkCSPSw 提取码: c5nh). If you have any question, please contact xf-liu19@mails.tsinghua.edu.cn.

```
+--1.raw_data
| +--plasma
| | +--AS_rawdata.txt
| | +--editing_totalcount_rawdata.txt
| | +--ASE_rawdata.txt
| | +--SNP_depth_sum.txt
| | +--ASE_depth_sum.txt
| | +--expression_rawdata.txt
| | +--editing_rawdata.txt
| | +--SNP_rowdata.txt
| | +--chimericRNA_rawdata.txt
| | +--SNP_alt_depth.txt
| | +--promoter_rowdata_TPM.txt
| | +--editing_altcount_rawdata.txt
| | +--ASE_alt_depth.txt
| +--platelet
| | +--chimericRNA_rawdata.txt
| | +--editing_totalcount_rawdata.txt
| | +--ASE_depth_sum.txt
| | +--AS_rawdata.txt
| | +--editing_rawdata.txt
| | +--SNP_depth_sum.txt
| | +--ASE_rawdata.txt
| | +--SNP_rawdata.txt
| | +--expression_rawdata.txt
| | +--ASE_alt_depth.txt
| | +--promoter_rawdata.txt
| | +--editing_altcount_rawdata.txt
| | +--SNP_alt_depth.txt
+--2.sample_id
| +--sample_cross_plasma.txt
| +--sample_cross_platelet.txt
+--3.data_gene_embedding
| +--embedding_all
| | +--plasma
| | | +--SNP
| | | | +--SNP_mean_window_3.txt
| | | | +--SNP_min.txt
| | | | +--SNP_mean_window_2.txt
| | | | +--SNP_mean.txt
| | | | +--SNP_count.txt
| | | | +--SNP_mean_all.txt
| | | | +--SNP_mean_window_1.txt
| | | | +--SNP_ID.txt
| | | | +--SNP_max.txt
| | | | +--SNP_entropy.txt
| | | +--AS
| | | | +--splicing_entropy_SE.txt
| | | | +--splicing_mean_RI.txt
| | | | +--splicing_max_A3SS.txt
| | | | +--splicing_mean_A3SS.txt
| | | | +--splicing_max_SE.txt
| | | | +--splicing_entropy_A5SS.txt
| | | | +--splicing_min_RI.txt
| | | | +--splicing_max_MXE.txt
| | | | +--splicing_entropy_MXE.txt
| | | | +--splicing_count_A3SS.txt
| | | | +--splicing_count_SE.txt
| | | | +--splicing_min_A3SS.txt
| | | | +--splicing_count_MXE.txt
| | | | +--splicing_count_RI.txt
| | | | +--splicing_min_A5SS.txt
| | | | +--splicing_count_A5SS.txt
| | | | +--splicing_min_MXE.txt
| | | | +--splicing_mean_SE.txt
| | | | +--splicing_mean_A5SS.txt
| | | | +--splicing_entropy_RI.txt
| | | | +--splicing_max_A5SS.txt
| | | | +--splicing_min_SE.txt
| | | | +--splicing_entropy_A3SS.txt
| | | | +--splicing_max_RI.txt
| | | | +--splicing_mean_MXE.txt
| | | +--expression
| | | | +--ALL_data_TPM.txt
| | | +--chimeric
| | | | +--chimeric_count.txt
| | | | +--chimeric_ID.txt
| | | +--editing
| | | | +--editing_ID.txt
| | | | +--editing_mean_window_2.txt
| | | | +--editing_mean_all.txt
| | | | +--editing_min.txt
| | | | +--editing_mean_window_3.txt
| | | | +--editing_max.txt
| | | | +--editing_entropy.txt
| | | | +--editing_mean.txt
| | | | +--editing_count.txt
| | | | +--editing_mean_window_1.txt
| | | +--ASE
| | | | +--ASE_entropy.txt
| | | | +--ASE_mean.txt
| | | | +--ASE_mean_window_3.txt
| | | | +--ASE_ID.txt
| | | | +--ASE_mean_window_2.txt
| | | | +--ASE_mean_all.txt
| | | | +--ASE_max.txt
| | | | +--ASE_mean_window_1.txt
| | | | +--ASE_min.txt
| | | | +--ASE_count.txt
| | | +--promoter
| | | | +--promoter_count.txt
| | | | +--promoter_min.txt
| | | | +--promoter_max.txt
| | | | +--promoter_mean.txt
| | +--platelet
| | | +--SNP
| | | | +--SNP_mean_window_2.txt
| | | | +--SNP_min.txt
| | | | +--SNP_mean_all.txt
| | | | +--SNP_ID.txt
| | | | +--SNP_mean_window_3.txt
| | | | +--SNP_entropy.txt
| | | | +--SNP_count.txt
| | | | +--SNP_mean.txt
| | | | +--SNP_mean_window_1.txt
| | | | +--SNP_max.txt
| | | +--promoter
| | | | +--promoter_min.txt
| | | | +--promoter_mean.txt
| | | | +--promoter_count.txt
| | | | +--promoter_max.txt
| | | +--AS
| | | | +--splicing_max_A3SS.txt
| | | | +--splicing_min_MXE.txt
| | | | +--splicing_entropy_A5SS.txt
| | | | +--splicing_max_RI.txt
| | | | +--splicing_entropy_RI.txt
| | | | +--splicing_mean_A5SS.txt
| | | | +--splicing_min_SE.txt
| | | | +--splicing_count_A5SS.txt
| | | | +--splicing_entropy_MXE.txt
| | | | +--splicing_count_MXE.txt
| | | | +--splicing_min_A3SS.txt
| | | | +--splicing_count_SE.txt
| | | | +--splicing_mean_SE.txt
| | | | +--splicing_count_A3SS.txt
| | | | +--splicing_mean_RI.txt
| | | | +--splicing_min_A5SS.txt
| | | | +--splicing_count_RI.txt
| | | | +--splicing_max_A5SS.txt
| | | | +--splicing_entropy_A3SS.txt
| | | | +--splicing_min_RI.txt
| | | | +--splicing_max_MXE.txt
| | | | +--splicing_mean_MXE.txt
| | | | +--splicing_mean_A3SS.txt
| | | | +--splicing_entropy_SE.txt
| | | | +--splicing_max_SE.txt
| | | +--expression
| | | | +--ALL_data_TPM.txt
| | | +--editing
| | | | +--editing_mean_window_1.txt
| | | | +--editing_count.txt
| | | | +--editing_entropy.txt
| | | | +--editing_mean_all.txt
| | | | +--editing_min.txt
| | | | +--editing_ID.txt
| | | | +--editing_max.txt
| | | | +--editing_mean_window_3.txt
| | | | +--editing_mean.txt
| | | | +--editing_mean_window_2.txt
| | | +--chimeric
| | | | +--chimeric_ID.txt
| | | | +--chimeric_count.txt
| | | +--ASE
| | | | +--ASE_max.txt
| | | | +--ASE_mean.txt
| | | | +--ASE_mean_window_1.txt
| | | | +--ASE_count.txt
| | | | +--ASE_ID.txt
| | | | +--ASE_entropy.txt
| | | | +--ASE_mean_window_3.txt
| | | | +--ASE_mean_all.txt
| | | | +--ASE_mean_window_2.txt
| | | | +--ASE_min.txt
| +--merge
| | +--plasma
| | | +--data_all.npy
| | +--platelet
| | | +--data_all.npy
```
