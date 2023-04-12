#!/bin/bash


mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/5.MOGOnet
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/5.MOGOnet/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/5.MOGOnet/${i}/${m}
for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/MOGONet/main.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype_new_final.txt \
--feature_num 1000 \
--dataset ${i} \
--data_type merge \
--stop 200 \
--num 2000 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/5.MOGOnet/${i}/${m}
done


mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/5.MOGOnet
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/5.MOGOnet/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/5.MOGOnet/${i}/${m}
for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/MOGONet/main.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage_new_final.txt \
--feature_num 1000 \
--dataset ${i} \
--data_type merge \
--stop 200 \
--num 2000 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/5.MOGOnet/${i}/${m}
done

mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/5.MOGOnet
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/5.MOGOnet/${i}
mkdir /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/5.MOGOnet/${i}/${m}
for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/MOGONet/main.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival_new_final.txt \
--feature_num 1000 \
--dataset ${i} \
--data_type merge \
--stop 200 \
--num 2000 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/5.MOGOnet/${i}/${m}
done

