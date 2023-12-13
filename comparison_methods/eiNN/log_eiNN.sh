#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/eiNN
cd ${root_path}/comparison_methods/eiNN
python3 ${root_path}/comparison_methods/eiNN/1.model.py \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1000/data_all.txt \
--dataset 1 \
--epoch_num 2000 \
--learning_rate 1-e5 \
--patience 10 \
--delta 1-e2 \
--stop_epoch 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/eiNN/


mkdir ${root_path}/comparison_methods/Result/BRCA_survival/eiNN
cd ${root_path}/comparison_methods/eiNN
python3 ${root_path}/comparison_methods/eiNN/1.model.py \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1000/data_all.txt \
--dataset 1 \
--epoch_num 2000 \
--learning_rate 1-e5 \
--patience 10 \
--delta 1-e2 \
--stop_epoch 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/eiNN/
