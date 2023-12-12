#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/5.MOGOnet
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/5.MOGOnet/merge
cd ${root_path}/comparison_methods/MOGONet
python3 ${root_path}/comparison_methods/MOGONet/main.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--feature_num 1000 \
--dataset 1 \
--data_type merge \
--stop 200 \
--num 2000 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/5.MOGOnet/merge


mkdir ${root_path}/comparison_methods/Result/BRCA_survival/5.MOGOnet
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/5.MOGOnet/merge
cd ${root_path}/comparison_methods/MOGONet
python3 ${root_path}/comparison_methods/MOGONet/main.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--feature_num 1000 \
--dataset 1 \
--data_type merge \
--stop 200 \
--num 2000 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/5.MOGOnet/merge
