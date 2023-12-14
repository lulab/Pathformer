# Result of Pathformer

The following shows the directory tree of result folder. Due to storage space limitations on Github, only part of data is shown, other data can be downloaded from these links (链接: https://pan.baidu.com/s/1LUSd-aHVJUZqpZZ43ic8ug 提取码: 3q2b). If you have any question, please contact xf-liu19@mails.tsinghua.edu.cn.

```
+--BRCA_stage
| +--label_test.txt
| +--samplename_test.txt
| +--data_test.npy
| +--modal_type_all.txt
| +--ckpt
| | +--BRCA_stage_best.pth
| +--Interpretability
| | +--gene_pathway.h5
| | +--attn_out_row_all.h5
| | +--attn_out_col_all.h5
| | +--data_label.npy
| | +--data_label_SHAP.npy
| | +--shap_pathway_all.h5
| | +--data_label_SHAP_pathway.npy
| | +--important_omics.txt
| | +--important_omics_pie.pdf
| | +--shap_important_pathway.txt
| | +--shap_important_pathway_top15.txt
| | +--shap_value_pathway.txt
| | +--shap_pathway_modal.txt
| | +--shap_important_pathway_modal.txt
| | +--shap_gene_modal.txt
| | +--shap_important_gene_modal.txt
| | +--pathway_crosstalk_network_update.npy
| | +--pathway_sub_network_score_all.txt
| | +--pathway_network_hub_modul_pathway.txt
| | +--pathway_network_hub_modul_weight.txt
| | +--shap_important_gene_modal_.txt
| | +--shap_gene_all.h5.tar.gz
| | +--net_all.h5.tar.gz
| +--predict_score.txt
| +--result.txt
| +--result_predict_evaluate.txt
+--BRCA_survival
| +--label_test.txt
| +--samplename_test.txt
| +--data_test.npy
| +--modal_type_all.txt
| +--ckpt
| | +--BRCA_survival_best.pth
| +--Interpretability
| | +--gene_pathway.h5
| | +--attn_out_col_all.h5
| | +--data_label.npy
| | +--important_omics.txt
| | +--important_omics_pie.pdf
| | +--data_label_SHAP.npy
| | +--shap_pathway_all.h5
| | +--data_label_SHAP_pathway.npy
| | +--shap_value_pathway.txt
| | +--shap_important_pathway_top15.txt
| | +--shap_pathway_modal.txt
| | +--shap_important_pathway_modal.txt
| | +--shap_gene_modal.txt
| | +--shap_important_gene_modal.txt
| | +--pathway_crosstalk_network_update.npy
| | +--pathway_sub_network_score_all.txt
| | +--pathway_network_hub_modul_pathway.txt
| | +--pathway_network_hub_modul_weight.txt
| | +--shap_gene_all.h5.tar.gz
| | +--net_all.h5.tar.gz
| | +--attn_out_row_all.h5.tar.gz
| +--predict_score.txt
| +--result.txt
| +--result_predict_evaluate.txt
+--plasma
| +--label_test.txt
| +--modal_type_all.txt
| +--ckpt
| | +--plasma_best.pth
| +--result.txt
| +--samplename_test.txt
| +--data_test.npy
| +--predict_score.txt
| +--result_predict_evaluate.txt
| +--Interpretability
| | +--gene_pathway.h5
| | +--attn_out_col_all.h5
| | +--data_label.npy
| | +--important_omics.txt
| | +--important_omics_pie.pdf
| | +--data_label_SHAP.npy
| | +--shap_pathway_all.h5
| | +--data_label_SHAP_pathway.npy
| | +--shap_value_pathway.txt
| | +--shap_important_pathway_top15.txt
| | +--shap_pathway_modal.txt
| | +--shap_important_pathway_modal.txt
| | +--shap_gene_modal.txt
| | +--shap_important_gene_modal.txt
| | +--pathway_crosstalk_network_update.npy
| | +--pathway_sub_network_score_all.txt
| | +--pathway_network_hub_modul_pathway.txt
| | +--pathway_network_hub_modul_weight.txt
| | +--shap_important_gene_modal_.txt
| | +--shap_gene_all.h5.tar.gz
| | +--net_all.h5.tar.gz
| | +--attn_out_row_all.h5.tar.gz
+--platelet
| +--modal_type_all.txt
| +--modal_select.txt
| +--ckpt
| | +--platelet_best.pth
| +--result.txt
```
