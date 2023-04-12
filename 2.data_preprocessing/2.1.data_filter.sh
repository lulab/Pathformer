#!/bin/bash

for num in 1 2 3 4 5 6 7 8 9 10
do
if [ ${num} -lt 2 ];then d="ib-node155";elif [ ${num} -lt 3 ];then d="ib-node162";elif [ ${num} -lt 4 ];then d="ib-node163"
elif [ ${num} -lt 5 ];then d="ib-node164";elif [ ${num} -lt 6 ];then d="ib-node165";elif [ ${num} -lt 7 ];then d="ib-node170"
elif [ ${num} -lt 8 ];then d="ib-node171";elif [ ${num} -lt 9 ];then d="ib-node173";elif [ ${num} -lt 10 ];then d="ib-node174"
else d="ib-node155";fi
#echo ${num}
echo -e "#!/bin/sh
#BSUB -J BRCA
#BSUB -o /err/output.%J.txt
#BSUB -e /err/err.%J.txt
#BSUB -q normal
#BSUB -n 5
#BSUB -m ${d}


python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/2.1.data_filter.py \\
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \\
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_subtype.txt \\
--feature_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/1.diff_feature_subtype/${num}/ \\
--feature_num_type ANOVA_1000 \\
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_subtype/${num}
">/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/log/ANOVA_1000_log_subtype_${num}.sh
done




for num in 1 2 3 4 5 6 7 8 9 10
do
if [ ${num} -lt 2 ];then d="ib-node155";elif [ ${num} -lt 3 ];then d="ib-node162";elif [ ${num} -lt 4 ];then d="ib-node163"
elif [ ${num} -lt 5 ];then d="ib-node164";elif [ ${num} -lt 6 ];then d="ib-node165";elif [ ${num} -lt 7 ];then d="ib-node170"
elif [ ${num} -lt 8 ];then d="ib-node171";elif [ ${num} -lt 9 ];then d="ib-node173";elif [ ${num} -lt 10 ];then d="ib-node174"
else d="ib-node155";fi
#echo ${num}
echo -e "#!/bin/sh
#BSUB -J BRCA
#BSUB -o /err/output.%J.txt
#BSUB -e /err/err.%J.txt
#BSUB -q normal
#BSUB -n 5
#BSUB -m ${d}

python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/2.1.data_filter.py \\
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \\
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_stage.txt \\
--feature_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/1.diff_feature_stage/${num}/ \\
--feature_num_type ANOVA_1000 \\
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_stage/${num}
">/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/log/ANOVA_1000_log_stage_${num}.sh
done




for num in 1 2 3 4 5 6 7 8 9 10
do
if [ ${num} -lt 2 ];then d="ib-node155";elif [ ${num} -lt 3 ];then d="ib-node162";elif [ ${num} -lt 4 ];then d="ib-node163"
elif [ ${num} -lt 5 ];then d="ib-node164";elif [ ${num} -lt 6 ];then d="ib-node165";elif [ ${num} -lt 7 ];then d="ib-node170"
elif [ ${num} -lt 8 ];then d="ib-node171";elif [ ${num} -lt 9 ];then d="ib-node173";elif [ ${num} -lt 10 ];then d="ib-node174"
else d="ib-node155";fi
#echo ${num}
echo -e "#!/bin/sh
#BSUB -J BRCA
#BSUB -o /err/output.%J.txt
#BSUB -e /err/err.%J.txt
#BSUB -q normal
#BSUB -n 5
#BSUB -m ${d}

python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/2.1.data_filter.py \\
--path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/ \\
--label_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/sample_id/sample_cross_survival.txt \\
--feature_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/1.diff_feature_survival/${num}/ \\
--feature_num_type ANOVA_1000 \\
--save_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/2.data_feature_survival/${num}
">/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/log/ANOVA_1000_log_survival_${num}.sh
done

