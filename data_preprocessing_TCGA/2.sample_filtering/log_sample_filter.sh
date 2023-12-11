#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

python3 ${root_path}/data_preprocessing_TCGA/2.sample_filtering/1.sample_all_filter.py \
--raw_data_path ${root_path}/data_TCGA/1.raw_data/ \
--save_path ${root_path}/data_TCGA/2.sample_id/ \
--cancer BRCA

python3 ${root_path}/data_preprocessing_TCGA/2.sample_filtering/2.1.sample_stage.py \
--stage_label_path ${root_path}/data/raw_data/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix \
--save_path ${root_path}/data_TCGA/2.sample_id/

python3 ${root_path}/data_preprocessing_TCGA/2.sample_filtering/2.2.sample_survival.py \
--survival_label_path ${root_path}/data/raw_data/survival_BRCA_survival.txt \
--save_path ${root_path}/data_TCGA/2.sample_id/


