#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do

Rscript /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/mixOmics/model_PLSDA.R \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/${i}/1000/data_count.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/${i}/1000/data_methylation.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/${i}/1000/data_CNV.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype.txt \
${i} \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_subtype/mixOmics/PLSDA/${i}/merge/

done

for i in 1 2 3 4 5 6 7 8 9 10
do

Rscript /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/mixOmics/model_PLSDA.R \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/${i}/1000/data_count.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/${i}/1000/data_methylation.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/${i}/1000/data_CNV.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage.txt \
${i} \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_stage/mixOmics/PLSDA/${i}/merge/

done

for i in 1 2 3 4 5 6 7 8 9 10
do

Rscript /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/3.comparison_methods/mixOmics/model_PLSDA.R \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/${i}/1000/data_count.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/${i}/1000/data_methylation.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/${i}/1000/data_CNV.txt \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival.txt \
${i} \
/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/Result/BRCA_survival/mixOmics/PLSDA/${i}/merge/

done
