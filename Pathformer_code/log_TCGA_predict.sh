#!/bin/sh

cd /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Pathformer_code
echo 'BRCA_stage'
python3 Pathformer_predict.py \
--modal_all_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/modal_type_all.txt \
--modal_select_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/3.data_gene_embedding/merge/merge_data_stage/data_all.npy \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/label_validation.txt \
--sample_name_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/samplename_validation.txt \
--model_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/ckpt/BRCA_stage_best.pth \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/ \
--label_dim 2 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'BRCA_subtype'
python3 Pathformer_predict.py \
--modal_all_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/3.data_gene_embedding/merge/merge_data_subtype/data_all.npy \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/label_validation.txt \
--sample_name_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/samplename_validation.txt \
--model_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/ckpt/BRCA_subtype_best.pth \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/ \
--label_dim 5 \
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
--modal_all_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/modal_type_all.txt \
--modal_select_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/data_all.npy \
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/label_validation.txt \
--sample_name_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/samplename_validation.txt \
--model_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/ckpt/BRCA_survival_best.pth \
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/Pathformer/ \
--label_dim 2 \
--evaluate True \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3
