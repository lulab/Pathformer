#!/bin/sh

for c in BRCA
do
echo ${c}
python3.8 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_gene_embedding/3.sample_all_filter.py \
--raw_data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/ \
--embedding_data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/raw/ \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/ \
--cancer ${c}
done