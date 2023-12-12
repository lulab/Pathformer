#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

echo "BRCA_stage"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method/LR
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method/SVM
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method/RF
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method/XGBoost

for m in LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 ${root_path}/3.comparison_methods/late_integration_method/model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1000/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--feature_name 1000 \
--dataset 1 \
--method ${m} \
--num 1 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/late_integration_method/
done
echo "BRCA_stage ok"

echo "BRCA_survival"
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method/LR
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method/SVM
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method/RF
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method/XGBoost

for m in LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 ${root_path}/3.comparison_methods/late_integration_method/model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1000/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--feature_name 1000 \
--dataset 1 \
--method ${m} \
--num 1 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/late_integration_method/
done
echo "BRCA_survival ok"
