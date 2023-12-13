# TCGA data

Here we take the breast cancer as an example, including the data of breast cancer early- and late- stage classification, breast cancer low- and high- survival risk classification, and breast cancer subtype classification. Breast cancer data preprocessing process can be seen in the folder: **/data_preprocessing_TCGA** and **/comparison_methods/data_feature_filter**.

The following shows the directory tree of data folder. Due to storage space limitations on Github, data can be  downloaded from these links (链接: https://pan.baidu.com/s/1_-BjFkLdJOiD0m53VuvxMg 提取码: mco2). If you have any question, please contact xf-liu19@mails.tsinghua.edu.cn.

```
+--1.raw_data
| +--BRCA.CNV_seg.csv
| +--BRCA.CNV_gene_level.txt
| +--TCGA.BRCA.sampleMap_BRCA_clinicalMatrix
| +--BRCA.DNAmethy.csv
| +--BRCA.DNAmethy.RData
| +--BRCA.CNV_seg.RData
| +--BRCA.CNV_masked_seg.RData
| +--BRCA.CNV_gene.RData
| +--survival_BRCA_survival.txt
| +--BRCA.CNV_masked_seg.csv
| +--BRCA.miRNA.csv
| +--BRCA.mRNA.RData
| +--BRCA.CNV_masked_seg_filter.txt
| +--BRCA.CNV_gene.csv
| +--BRCA.miRNA.RData
| +--BRCA.mRNA.csv
+--2.sample_id
| +--sample_miRNA_data_2.txt
| +--sample_CNV_data_2.txt
| +--sample_CNV_gene_data_2.txt
| +--sample_mRNA_data_2.txt
| +--sample_DNA_data_2.txt
| +--sample_id_label_2.txt
| +--sample_id_label_3.txt
| +--sample_survival.txt
| +--sample_cross_survival.txt
| +--sample_stage.txt
| +--sample_cross_stage.txt
+--3.data_gene_embedding
| +--embedding_raw
| | +--promoter_methylation_max.txt
| | +--CNV_mean.txt
| | +--methylation_count.txt
| | +--CNV_min.txt
| | +--DNA_methylation_promoterid.txt
| | +--promoter_methylation_mean.txt
| | +--methylation_mean.txt
| | +--methylation_max.txt
| | +--methylation_min.txt
| | +--CNV_count.txt
| | +--promoter_methylation_count.txt
| | +--DNA_methylation_geneid.txt
| | +--promoter_methylation_min.txt
| | +--CNV_max.txt
| | +--CNV_id.txt
| +--embedding_all
| | +--RNA_all_rawdata.txt
| | +--methylation_mean.txt
| | +--promoter_methylation_max.txt
| | +--miRNA_rawdata.txt
| | +--promoter_methylation_count.txt
| | +--CNV_max.txt
| | +--RNA_all_TPM.txt
| | +--methylation_min.txt
| | +--methylation_count.txt
| | +--promoter_methylation_mean.txt
| | +--mRNA_rawdata.txt
| | +--CNV_min.txt
| | +--CNV_gene_level.txt
| | +--CNV_masked_rawdata.txt
| | +--methylation_max.txt
| | +--CNV_mean.txt
| | +--CNV_count.txt
| | +--methylation_rawdata.txt
| | +--promoter_methylation_min.txt
| +--merge
| | +--merge_data_stage
| | | +--data_all.npy
| | +--merge_data_survival
| | | +--data_all.npy
+--4.data_feature_filter_of_comparison_methods
| +--1.diff_feature_stage
| | +--filter_feature_ANOVA_100_methylation_mean.txt
| | +--filter_feature_ANOVA_100_methylation_min.txt
| | +--filter_feature_ANOVA_100_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_100_CNV_gene_level.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_all_methylation_mean.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_count.txt
| | +--filter_feature_CNV_max.txt
| | +--filter_feature_ANOVA_all_CNV_max.txt
| | +--filter_feature_ANOVA_all_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_all_methylation_count.txt
| | +--filter_feature_ANOVA_1000_CNV_max.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_1000_CNV_mean.txt
| | +--filter_feature_ANOVA_500_CNV_max.txt
| | +--filter_feature_ANOVA_500_methylation_max.txt
| | +--filter_feature_ANOVA_100_CNV_count.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_all_methylation_max.txt
| | +--filter_feature_ANOVA_1000_CNV_count.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_100_CNV_min.txt
| | +--filter_feature_promoter_methylation_max.txt
| | +--filter_feature_CNV_mean.txt
| | +--filter_feature_methylation_max.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_1000_methylation_min.txt
| | +--filter_feature_ANOVA_100_CNV_mean.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_count.txt
| | +--filter_feature_promoter_methylation_mean.txt
| | +--filter_feature_CNV_gene_level.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_500_methylation_min.txt
| | +--filter_feature_methylation_mean.txt
| | +--filter_feature_ANOVA_500_CNV_min.txt
| | +--filter_feature_methylation_count.txt
| | +--filter_feature_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_1000_CNV_min.txt
| | +--filter_feature_ANOVA_1000_methylation_count.txt
| | +--filter_feature_CNV_count.txt
| | +--filter_feature_ANOVA_1000_methylation_max.txt
| | +--filter_feature_methylation_min.txt
| | +--filter_feature_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_100_CNV_max.txt
| | +--filter_feature_ANOVA_1000_methylation_mean.txt
| | +--filter_feature_ANOVA_500_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_1000_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_all_methylation_min.txt
| | +--filter_feature_ANOVA_500_CNV_gene_level.txt
| | +--filter_feature_ANOVA_all_CNV_count.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_all_CNV_gene_level.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_mean.txt
| | +--filter_feature_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_1000_CNV_gene_level.txt
| | +--filter_feature_ANOVA_100_methylation_max.txt
| | +--filter_feature_ANOVA_500_CNV_mean.txt
| | +--filter_feature_ANOVA_500_CNV_count.txt
| | +--filter_feature_ANOVA_500_methylation_mean.txt
| | +--filter_feature_ANOVA_all_CNV_mean.txt
| | +--filter_feature_ANOVA_500_methylation_count.txt
| | +--filter_feature_ANOVA_100_methylation_count.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_all_CNV_min.txt
| | +--filter_feature_CNV_min.txt
| +--1.diff_feature_survival
| | +--filter_feature_ANOVA_all_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_100_CNV_max.txt
| | +--filter_feature_ANOVA_all_CNV_count.txt
| | +--filter_feature_ANOVA_100_CNV_gene_level.txt
| | +--filter_feature_ANOVA_500_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_1000_methylation_mean.txt
| | +--filter_feature_ANOVA_500_CNV_mean.txt
| | +--filter_feature_methylation_max.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_all_CNV_mean.txt
| | +--filter_feature_CNV_min.txt
| | +--filter_feature_ANOVA_1000_CNV_gene_level.txt
| | +--filter_feature_ANOVA_100_methylation_max.txt
| | +--filter_feature_ANOVA_500_CNV_count.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_1000_methylation_max.txt
| | +--filter_feature_methylation_count.txt
| | +--filter_feature_ANOVA_500_CNV_min.txt
| | +--filter_feature_ANOVA_all_methylation_min.txt
| | +--filter_feature_ANOVA_100_methylation_count.txt
| | +--filter_feature_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_1000_CNV_min.txt
| | +--filter_feature_ANOVA_500_methylation_count.txt
| | +--filter_feature_ANOVA_all_CNV_min.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_1000_CNV_count.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_mean.txt
| | +--filter_feature_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_1000_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_500_methylation_min.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_500_methylation_mean.txt
| | +--filter_feature_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_all_methylation_mean.txt
| | +--filter_feature_ANOVA_1000_CNV_mean.txt
| | +--filter_feature_ANOVA_all_methylation_count.txt
| | +--filter_feature_ANOVA_all_CNV_max.txt
| | +--filter_feature_ANOVA_1000_CNV_max.txt
| | +--filter_feature_ANOVA_all_RNA_all_TPM.txt
| | +--filter_feature_ANOVA_all_methylation_max.txt
| | +--filter_feature_ANOVA_100_methylation_mean.txt
| | +--filter_feature_ANOVA_100_CNV_count.txt
| | +--filter_feature_ANOVA_500_CNV_gene_level.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_max.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_mean.txt
| | +--filter_feature_ANOVA_500_methylation_max.txt
| | +--filter_feature_ANOVA_100_CNV_mean.txt
| | +--filter_feature_methylation_mean.txt
| | +--filter_feature_ANOVA_100_RNA_all_TPM.txt
| | +--filter_feature_promoter_methylation_min.txt
| | +--filter_feature_RNA_all_TPM.txt
| | +--filter_feature_methylation_min.txt
| | +--filter_feature_ANOVA_1000_methylation_count.txt
| | +--filter_feature_CNV_mean.txt
| | +--filter_feature_ANOVA_all_CNV_gene_level.txt
| | +--filter_feature_CNV_gene_level.txt
| | +--filter_feature_ANOVA_100_CNV_min.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_500_CNV_max.txt
| | +--filter_feature_ANOVA_1000_methylation_min.txt
| | +--filter_feature_ANOVA_500_promoter_methylation_max.txt
| | +--filter_feature_CNV_count.txt
| | +--filter_feature_ANOVA_100_methylation_min.txt
| | +--filter_feature_CNV_max.txt
| | +--filter_feature_ANOVA_1000_promoter_methylation_min.txt
| | +--filter_feature_ANOVA_all_promoter_methylation_count.txt
| | +--filter_feature_ANOVA_100_promoter_methylation_max.txt
| +--2.data_feature_stage
| | +--1000
| | | +--data_count.txt
| | | +--data_methylation.txt
| | | +--data_CNV.txt
| | | +--data_all.txt
| +--2.data_feature_survival
| | +--1000
| | | +--data_all.txt
| | | +--data_count.txt
| | | +--data_CNV.txt
| | | +--data_methylation.txt
| +--3.merge_all_stage
| | +--data_count_all.txt
| | +--data_methylation_all.txt
| | +--data_CNV_all.txt
| +--3.merge_all_stage_pca
| | +--data_methylation_validation_all.txt
| | +--data_CNV_test_all.txt
| | +--data_count_train_all.txt
| | +--data_count_validation_all.txt
| | +--data_CNV_train_all.txt
| | +--data_methylation_train_all.txt
| | +--data_methylation_test_all.txt
| | +--data_CNV_validation_all.txt
| | +--data_count_test_all.txt
| +--3.merge_all_survival
| | +--data_count_all.txt
| | +--data_CNV_all.txt
| | +--data_methylation_all.txt
| +--3.merge_all_survival_pca
| | +--data_count_train_all.txt
| | +--data_CNV_validation_all.txt
| | +--data_methylation_train_all.txt
| | +--data_count_test_all.txt
| | +--data_methylation_test_all.txt
| | +--data_methylation_validation_all.txt
| | +--data_CNV_train_all.txt
| | +--data_count_validation_all.txt
| | +--data_CNV_test_all.txt
```
