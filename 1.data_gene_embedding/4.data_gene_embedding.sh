#!/bin/sh
#BSUB -J BRCA
#BSUB -o output.%J.txt
#BSUB -e err.%J.txt
#BSUB -q normal
#BSUB -n 20
#BSUB -R \"span[hosts=1]\"

python3.8 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_gene_embedding/4.data_gene_embedding.py \
--raw_data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/ \
--embedding_data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/ \
--sample_id_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/ \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \
--cancer BRCA
