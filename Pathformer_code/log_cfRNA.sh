#!/bin/sh

cd /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper//Pathformer_code
python3 Pathformer_main.py \
--modal_all_path ../Result/plasma/modal_type_all.txt \
--modal_select_path ../Result/plasma/modal_type_all.txt \
--gene_all ../reference/gene_mRNA_cfRNA.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../data_cfRNA/3.data_gene_embedding/merge/plasma/data_all.npy \
--label_path ../data_cfRNA/2.sample_id/sample_cross_plasma.txt \
--save_path ../Result/plasma/ \
--dataset 1 \
--model_name plasma \
--model_save True \
--batch_size 8 \
--gradient_num 4 \
--epoch_num 2000 \
--early_stopping_type f1_macro_2 \
--patience 10 \
--delta 1e-2 \
--stop_epoch 100 \
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
--modal_all_path ../Result/platelet/modal_type_all.txt \
--modal_select_path ../Result/platelet/modal_select.txt \
--gene_all ../reference/gene_mRNA_cfRNA.txt \
--gene_select ../reference/Pathformer_select_gene.txt \
--pathway_gene_w ../reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  ../reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--data_path ../data_cfRNA/3.data_gene_embedding/merge/platelet/data_all.npy \
--label_path ../data_cfRNA/2.sample_id/sample_cross_platelet_new_final.txt \
--save_path ../Result/platelet/ \
--dataset 1 \
--model_name platelet \
--model_save True \
--batch_size 8 \
--gradient_num 4 \
--epoch_num 2000 \
--early_stopping_type f1_macro_2 \
--patience 10 \
--delta 1e-2 \
--stop_epoch 100 \
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
