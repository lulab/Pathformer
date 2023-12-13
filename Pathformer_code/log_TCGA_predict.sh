#!/bin/sh

cd /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Pathformer_code
echo 'BRCA_stage'
python3 Pathformer_predict.py \
--modal_all_path ../Result/BRCA_stage/modal_type_all.txt \
--modal_select_path ../Result/BRCA_stage/modal_type_all.txt \
--gene_all ../reference/gene_all.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../Result/BRCA_stage/data_test.npy \
--label_path ../Result/BRCA_stage/label_test.txt \
--sample_name_path ../Result/BRCA_stage/samplename_test.txt \
--model_path ../Result/BRCA_stage/ckpt/BRCA_stage_best.pth \
--save_path ../Result/BRCA_stage/ \
--label_dim 2 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'BRCA_survival'
python3 Pathformer_predict.py \
--modal_all_path ../Result/BRCA_survival/modal_type_all.txt \
--modal_select_path ../Result/BRCA_survival/modal_type_all.txt \
--gene_all ../reference/gene_all.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../Result/BRCA_survival/data_test.npy \
--label_path ../Result/BRCA_survival/label_test.txt \
--sample_name_path ../Result/BRCA_survival/samplename_test.txt \
--model_path ../Result/BRCA_survival/ckpt/BRCA_survival_best.pth \
--save_path ../Result/BRCA_survival/ \
--label_dim 2 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3
