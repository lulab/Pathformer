#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

for d in plasma EV platelet
do
python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/1.data_expression_TPM.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/expression/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/2.data_AS.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/AS/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/3.1.data_ASE_ID.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/ASE/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/3.2.data_ASE.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/ASE/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/4.1.data_chimeric_ID.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/chimeric/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/4.2.data_chimeric.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/chimeric/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/5.1.data_editing_ID.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/editing/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/5.2.data_editing.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/editing/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/6.data_promoter.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/promoter/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/7.1.data_SNP_ID.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/SNP/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/7.2.data_SNP.py \
--input_path ${root_path}/data_cfRNA/1.raw_data/${d}/ \
--reference_path ${root_path}/reference/ \
--save_path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/SNP/

python3 ${root_path}/data_preprocessing_cfRNA/1.data_gene_embedding/8.data_merge.py \
--path ${root_path}/data_cfRNA/3.data_gene_embedding/embedding_all/${d}/ \
--reference_path ${root_path}/reference/ \
--label_path ${root_path}/data_cfRNA/data_gene_embedding/sample_id/sample_cross_${d}.txt \
--save_path ${root_path}/data_cfRNA/data_gene_embedding/merge/${d}/


done