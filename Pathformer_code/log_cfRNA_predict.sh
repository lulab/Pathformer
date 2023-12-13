#!/bin/sh

cd /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper//Pathformer_code
echo 'plasma'
python3 Pathformer_predict.py \
--modal_all_path ../Result/plasma/modal_type_all.txt \
--modal_select_path ../Result/plasma/modal_type_all.txt \
--gene_all ../reference/gene_mRNA_cfRNA.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../Result/plasma/data_test.npy \
--label_path ../Result/plasma/label_test.txt \
--sample_name_path ../Result/plasma/samplename_test.txt \
--model_path ../Result/plasma/ckpt/plasma_best.pth \
--save_path ../Result/plasma/ \
--label_dim 6 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'platelet'
python3 Pathformer_predict.py \
--modal_all_path ../Result/platelet/modal_type_all.txt \
--modal_select_path ../Result/platelet/modal_type_all.txt \
--gene_all ../reference/gene_mRNA_cfRNA.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../Result/platelet/data_test.npy \
--label_path ../Result/platelet/label_test.txt \
--sample_name_path ../Result/platelet/samplename_test.txt \
--model_path ../Result/platelet/ckpt/plasma_best.pth \
--save_path ../Result/platelet/ \
--label_dim 6 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3
