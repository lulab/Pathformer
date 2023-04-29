#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
echo "BRCA_stage"
python3 ${root_path}/comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset 1 \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage_new_final.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage_final/
python3 ${root_path}/comparison_methods/PathCNN/2.model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage_new_final.txt \
--model_name BRCA \
--dataset 1 \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/PathCNN/1/merge/
echo "BRCA_stage ok"

echo "BRCA_survival"
python3 ${root_path}/comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset 1 \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival_new_final.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival_final/
python3 ${root_path}/comparison_methods/PathCNN/2.model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival_new_final.txt \
--model_name BRCA \
--dataset 1 \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/PathCNN/1/merge/
echo "BRCA_survival ok"

echo "BRCA_subtype"
python3 ${root_path}/comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset 1 \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype_new_final.txt \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_subtype/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_subtype_final/
python3 ${root_path}/comparison_methods/PathCNN/2.model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_subtype/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype_new_final.txt \
--model_name BRCA \
--dataset 1 \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path ${root_path}/comparison_methods/Result/BRCA_subtype/PathCNN/1/merge/
echo "BRCA_subtype ok"