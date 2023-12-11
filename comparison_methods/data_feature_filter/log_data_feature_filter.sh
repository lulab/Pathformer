#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"
echo "feature_filter"
feature_path=${root_path}/reference/feature_type.txt

for i in `cat ${feature_path}`
do
echo ${i}
python3 ${root_path}/comparison_methods/data_feature_filter/1.1.feature_filter_of_ratio_0.py \
--feature_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/${i}.txt \
--path_label ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--savepath ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_stage/ \
--dataset 1 \
--p 0.1 \
--feature_type ${i}
done

for i in `cat ${feature_path}`
do
echo ${i}
python3 ${root_path}/comparison_methods/data_feature_filter/1.1.feature_filter_of_ratio_0.py \
--feature_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/${i}.txt \
--path_label ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--savepath ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_survival/ \
--dataset 1 \
--p 0.1 \
--feature_type ${i}
done

for i in `cat ${feature_path}`
do
echo ${i}
python3 ${root_path}/comparison_methods/data_feature_filter/1.2.feature_filter_ANOVA.py \
--feature_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/${i}.txt \
--path_label ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--savepath ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_stage/ \
--dataset 1 \
--scaler 1 \
--feature_type ${i}
done

for i in `cat ${feature_path}`
do
echo ${i}
python3 ${root_path}/comparison_methods/data_feature_filter/1.2.feature_filter_ANOVA.py \
--feature_path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/${i}.txt \
--path_label ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--savepath ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_survival/ \
--dataset 1 \
--scaler 1 \
--feature_type ${i}
done
echo "feature_filter ok"

echo "data_filter"

python3 ${root_path}/comparison_methods/data_feature_filter/2.1.data_filter.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--feature_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_stage/ \
--feature_num_type ANOVA_1000 \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_stage/

python3 ${root_path}/comparison_methods/data_feature_filter/2.1.data_filter.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--feature_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/1.diff_feature_survival/ \
--feature_num_type ANOVA_1000 \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/2.data_feature_survival/

echo "data_filter ok"

echo "data_merge"

python3 ${root_path}/data_preprocessing_TCGA/2.feature_filter/3.1.data_merge_all.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_stage.txt \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_stage/

python3 ${root_path}/data_preprocessing_TCGA/2.feature_filter/3.1.data_merge_all.py \
--path ${root_path}/data_TCGA/3.data_gene_embedding/embedding_all/ \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_survival.txt \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_TCGA/4.data_feature_filter_of_comparison_methods/3.merge_all_survival/

echo "data_merge ok"
