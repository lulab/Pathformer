#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/1.1.DNA_methylation_promoterid.py \
--reference_path ${root_path}/reference/ \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/1.2.DNA_methylation.py \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/1.3.DNA_methylation_promoter.py \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/2.1.CNV_id.py \
--reference_path ${root_path}/reference/ \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/2.2.CNV_data.py \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/2.3.CNV_data_merge.py \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/


python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/3.1.data_gene_embedding_all.py \
--rawdata_path ${root_path}/data_TCGA/1.raw_data/ \
--embedding_data_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_raw/ \
--reference_path ${root_path}/reference/ \
--sample_id_path ${root_path}/data_TCGA/2.sample_id/ \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/3.2.data_gene_embedding_merge.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--reference_path ${root_path}/reference/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_stage.txt \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_stage/

python3 ${root_path}/data_preprocessing_TCGA/3.data_gene_embedding/3.2.data_gene_embedding_merge.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--reference_path ${root_path}/reference/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_survival.txt \
--save_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_survival/
