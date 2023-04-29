#!/bin/bash

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

echo "BRCA_stage"
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1/KNN
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1/LR
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1/SVM
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1/RF
mkdir ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/1/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 ${root_path}/3.comparison_methods/early_integration_method/model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/1/1000/data_all.txt \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--feature_name 1000 \
--dataset 1 \
--method ${m} \
--num 1 \
--save_path ${root_path}/comparison_methods/Result/BRCA_stage/early_integration_method/
done
echo "BRCA_stage ok"

echo "BRCA_survival"
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1/KNN
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1/LR
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1/SVM
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1/RF
mkdir ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/1/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 ${root_path}/3.comparison_methods/early_integration_method/model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/1/1000/data_all.txt \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--feature_name 1000 \
--dataset 1 \
--method ${m} \
--num 1 \
--save_path ${root_path}/comparison_methods/Result/BRCA_survival/early_integration_method/
done
echo "BRCA_survival ok"

echo "BRCA_subtype"
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1/KNN
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1/LR
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1/SVM
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1/RF
mkdir ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/1/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 ${root_path}/3.comparison_methods/early_integration_method/model.py \
--data_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_subtype/1/1000/data_all.txt \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype.txt \
--feature_name 1000 \
--dataset 1 \
--method ${m} \
--num 1 \
--save_path ${root_path}/comparison_methods/Result/BRCA_subtype/early_integration_method/
done
echo "BRCA_subtype ok"
