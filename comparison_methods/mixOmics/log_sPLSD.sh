#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
Rscript ${root_path}/comparison_methods/mixOmics/model_sPLSD.R \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_subtype/1/1000/data_count.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_subtype/1/1000/data_methylation.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_subtype/1/1000/data_CNV.txt \
${root_path}/data_TCGA/2.sample_id/sample_cross_subtype.txt \
1 \
${root_path}/comparison_methods/Result/BRCA_subtype/mixOmics/sPLSDA/1/merge/


Rscript ${root_path}/comparison_methods/mixOmics/model_sPLSD.R \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1/1000/data_count.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1/1000/data_methylation.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1/1000/data_CNV.txt \
${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
1 \
${root_path}/comparison_methods/Result/BRCA_stage/mixOmics/sPLSDA/1/merge/


Rscript ${root_path}/comparison_methods/mixOmics/model_sPLSD.R \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1/1000/data_count.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1/1000/data_methylation.txt \
${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1/1000/data_CNV.txt \
${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
1 \
${root_path}/comparison_methods/Result/BRCA_survival/mixOmics/sPLSDA/1/merge/

python3 ${root_path}/comparison_methods/mixOmics/result_sPLSDA.py \
--result_path ${root_path}/comparison_methods/Result/BRCA_subtype/mixOmics/sPLSDA/1/merge/
