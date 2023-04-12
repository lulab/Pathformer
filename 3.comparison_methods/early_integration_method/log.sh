#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
echo ${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}/KNN
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}/LR
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}/SVM
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}/RF
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/${i}/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/early_integration_method/model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/${i}/1000/data_all.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype.txt \
--feature_name 1000 \
--dataset ${i} \
--method ${m} \
--num 1 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/early_integration_method/

done
done

for i in 1 2 3 4 5 6 7 8 9 10
do
echo ${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}/KNN
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}/LR
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}/SVM
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}/RF
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/${i}/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/early_integration_method/model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/${i}/1000/data_all.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage.txt \
--feature_name 1000 \
--dataset ${i} \
--method ${m} \
--num 1 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/early_integration_method/
done
done

for i in 1 2 3 4 5 6 7 8 9 10
do
echo ${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}/KNN
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}/LR
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}/SVM
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}/RF
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/${i}/XGBoost

for m in KNN_cv LR_cv RF_cv SVM_cv XGBoost_cv
do
echo ${m}
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/early_integration_method/model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/${i}/1000/data_all.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival.txt \
--feature_name 1000 \
--dataset ${i} \
--method ${m} \
--num 1 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/early_integration_method/
done
done
