#!/bin/sh

root_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/"

cd ${root_path}/Pathformer_code
echo 'attention_map'
python3 ${root_path}/code/Interpretability/1.attention_map.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--model_path ${root_path}/Result/BRCA_subtype/ckpt/BRCA_subtype_best.pth \
--data_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_subtype/data_all.npy \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype_new_final.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/ \
--dataset 1 \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'attention_weight'
python3 ${root_path}/code/Interpretability/1.attention_weight.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/

echo 'SHAP_gene_sample'
python3 ${root_path}/code/Interpretability/2.SHAP_gene_sample.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--model_path ${root_path}/Result/BRCA_subtype/ckpt/BRCA_subtype_best.pth \
--data_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_subtype/data_all.npy \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype_new_final.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/ \
--dataset 1 \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'SHAP_pathway_sample'
python3 ${root_path}/code/Interpretability/2.SHAP_pathway_sample.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--gene_all /apps/home/lulab_liuxiaofan/qhsky/project/reference/gene_all.txt \
--gene_select /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_select_gene.txt \
--pathway_gene_w /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_gene_weight.npy \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--model_path ${root_path}/Result/BRCA_subtype/ckpt/BRCA_subtype_best.pth \
--data_path ${root_path}/data_TCGA/3.data_gene_embedding/merge/merge_data_subtype/data_all.npy \
--label_path ${root_path}/data_TCGA/2.sample_id/sample_cross_subtype_new_final.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/ \
--dataset 1 \
--depth 3 \
--heads  8 \
--dim_head 32 \
--beta 1 \
--attn_dropout 0.2 \
--ff_dropout 0.2 \
--classifier_dropout 0.3

echo 'SHAP_pathway_modal'
python3 ${root_path}/code/Interpretability/2.SHAP_pathway_modal.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--pathway_name_path ${root_path}/reference/Pathformer_pathway.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/

echo 'SHAP_gene_modal'
python3 ${root_path}/code/Interpretability/2.SHAP_gene_modal.py \
--modal_all_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--modal_select_path ${root_path}/Result/BRCA_subtype/modal_type_all.txt \
--pathway_name_path ${root_path}/reference/Pathformer_pathway.txt \
--gene_name_path ${root_path}/reference/Pathformer_select_gene.txt \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/

echo 'pathway_sub_network_score'
python3 ${root_path}/code/Interpretability/3.pathway_sub_network_score.py \
--pathway_name_path ${root_path}/reference/Pathformer_pathway.txt \
--pathway_crosstalk_network  /apps/home/lulab_liuxiaofan/qhsky/project/reference/Pathformer_pathway_crosstalk_network_matrix.npy \
--save_path ${root_path}/Result/BRCA_subtype/Interpretability/ \
--cut_off 0.997