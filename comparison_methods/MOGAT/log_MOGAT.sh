#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/MOGAT
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/MOGAT/merge
cd ${root_path}/comparison_methods/MOGAT
python3 ${root_path}/comparison_methods/MOGAT/main.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1000/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--dataset 1 \
--stop_epoch 200 \
--epoch_num 2000 \
--delta 1-e2 \
--patience 10 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/MOGAT/


mkdir ${root_path}/comparison_methods/Result/BRCA_survival/MOGAT
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/MOGAT/merge
cd ${root_path}/comparison_methods/MOGAT
python3 ${root_path}/comparison_methods/MOGAT/main.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1000/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--dataset 1 \
--stop_epoch 200 \
--epoch_num 2000 \
--delta 1-e2 \
--patience 10 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/MOGAT/

