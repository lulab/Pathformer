#!/bin/sh

cd /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Pathformer_code
python3 Pathformer_main.py \
--modal_all_path ../Result/BRCA_stage/modal_type_all.txt \
--modal_select_path ../Result/BRCA_stage/modal_type_all.txt \
--gene_all ../reference/gene_all.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../data_TCGA/3.data_gene_embedding/merge/merge_data_stage/data_all.npy \
--label_path ../data_TCGA/2.sample_id/sample_cross_stage.txt \
--save_path ../Result/BRCA_stage/ \
--dataset 1 \
--model_name BRCA_stage \
--model_save True \
--batch_size 16 \
--gradient_num 3 \
--epoch_num 2000 \
--early_stopping_type f1_macro_2 \
--patience 10 \
--delta 1e-2 \
--stop_epoch 0 \
--validation_each_epoch_no 1 \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3 \
--lr_max 1e-5 \
--lr_min 1e-8


python3 Pathformer_main.py \
--modal_all_path ../Result/BRCA_survival/modal_type_all.txt \
--modal_select_path ../Result/BRCA_survival/modal_type_all.txt \
--gene_all ../reference/gene_all.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../data_TCGA/3.data_gene_embedding/merge/merge_data_survival/data_all.npy \
--label_path ../data_TCGA/2.sample_id/sample_cross_survival.txt \
--save_path ../Result/BRCA_survival/ \
--dataset 1 \
--model_name BRCA_survival \
--model_save True \
--batch_size 16 \
--gradient_num 3 \
--epoch_num 2000 \
--early_stopping_type f1_macro_2 \
--patience 10 \
--delta 1e-2 \
--stop_epoch 0 \
--validation_each_epoch_no 1 \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3 \
--lr_max 1e-5 \
--lr_min 1e-8
