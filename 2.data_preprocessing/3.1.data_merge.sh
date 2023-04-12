#!/bin/sh
#BSUB -J BRCA
#BSUB -o /err/output.%J.txt
#BSUB -e /err/err.%J.txt
#BSUB -q normal
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -m ib-node166

python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/3.1.data_merge.py \
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \
--feature_type_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/feature_type.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage.txt \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/3.merge_data_stage/



python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/3.1.data_merge.py \
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \
--feature_type_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/feature_type.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype.txt \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/3.merge_data_subtype/


python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/3.1.data_merge.py \
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \
--feature_type_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/feature_type.txt \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival.txt \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/3.merge_data_survival/