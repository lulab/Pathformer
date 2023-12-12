#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
echo "BRCA_stage"
python3 ${root_path}/comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset 1 \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage_new_final.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage_pca/

python3 ${root_path}/comparison_methods/PathCNN/2.model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage_pca/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--model_name BRCA \
--dataset 1 \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/PathCNN/merge/
echo "BRCA_stage ok"

echo "BRCA_survival"
python3 ${root_path}/comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset 1 \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival_new_final.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival_pca/

python3 ${root_path}/comparison_methods/PathCNN/2.model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival_pca/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--model_name BRCA \
--dataset 1 \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/PathCNN/merge/
echo "BRCA_survival ok"
