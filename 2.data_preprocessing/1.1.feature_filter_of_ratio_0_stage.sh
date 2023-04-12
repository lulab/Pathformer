#!/bin/bash

feature_path=$"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/feature_type.txt"

for c in BRCA
do
for i in `cat ${feature_path}`
do
echo ${i}
for num in 1 2 3 4 5 6 7 8 9 10
do
if [ ${num} -lt 2 ];then d="ib-node155";elif [ ${num} -lt 3 ];then d="ib-node162";elif [ ${num} -lt 4 ];then d="ib-node163"
elif [ ${num} -lt 5 ];then d="ib-node164";elif [ ${num} -lt 6 ];then d="ib-node165";elif [ ${num} -lt 7 ];then d="ib-node170"
elif [ ${num} -lt 8 ];then d="ib-node171";elif [ ${num} -lt 9 ];then d="ib-node173";elif [ ${num} -lt 10 ];then d="ib-node174"
else d="ib-node155";fi
#echo ${num}
echo -e "#!/bin/sh
#BSUB -J ${c}
#BSUB -o /err/output.%J.txt
#BSUB -e /err/err.%J.txt
#BSUB -q normal
#BSUB -n 5
#BSUB -m ${d}

python3 /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/1.1.feature_filter_of_ratio_0.py \\
--feature_path /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_gene_embedding/embedding/${i}.txt \\
--path_label /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_data/TCGA_new/${c}/sample_id/sample_cross_stage.txt \\
--savepath /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/data_preprocessing/1.diff_feature_stage/${num}/ \\
--dataset ${num} \\
--p 0.1 \\
--feature_type ${i}
">/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/2.data_preprocessing/log/${i}_feature_filter_of_ratio_0_stage_${num}.sh
done
done
done