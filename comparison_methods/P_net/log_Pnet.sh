#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/P_net
cd ${root_path}/comparison_methods/P_net
python3 ${root_path}/comparison_methods/P_net/1.model.py \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--data_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_stage/ \
--dataset 1 \
--epoch_num 2000 \
--learning_rate 1-e5 \
--patience 10 \
--delta 1-e2 \
--stop_epoch 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/P_net/


mkdir ${root_path}/comparison_methods/Result/BRCA_survival/P_net
cd ${root_path}/comparison_methods/P_net
python3 ${root_path}/comparison_methods/P_net/1.model.py \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--data_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_survival/ \
--dataset 1 \
--epoch_num 2000 \
--learning_rate 1-e5 \
--patience 10 \
--delta 1-e2 \
--stop_epoch 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/P_net/
