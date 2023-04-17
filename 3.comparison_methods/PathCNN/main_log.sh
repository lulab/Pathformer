#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset ${i} \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype_new_final.txt \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_subtype/ \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_subtype_final/


python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/2.model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_subtype/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype_new_final.txt \
--model_name BRCA \
--dataset ${i} \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/PathCNN/${i}/${m}/
done

for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset ${i} \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage_new_final.txt \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_stage/ \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_stage_final/


python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/2.model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_stage/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage_new_final.txt \
--model_name BRCA \
--dataset ${i} \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/PathCNN/${i}/${m}/
done

for i in 1 2 3 4 5 6 7 8 9 10
do
python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/1.omics_image_PCA.py \
--dataset ${i} \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival_new_final.txt \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_survival/ \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_survival_final/


python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/PathCNN/2.model.py \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/4.merge_all_survival/ \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival_new_final.txt \
--model_name BRCA \
--dataset ${i} \
--data_type merge \
--epoch_num 2000 \
--learning_rate 1e-5 \
--stop 200 \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/PathCNN/${i}/${m}/
done