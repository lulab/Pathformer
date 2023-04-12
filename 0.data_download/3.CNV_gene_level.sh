#!/bin/sh

gistic2 \
-b /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/gistic2/BRCA \
-seg  /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data/raw_data/BRCA.CNV_masked_seg_filter.txt \
-mk /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/snp6.na35.remap.hg38.subset.marker_file.txt \
-refgene /apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/reference/hg38.UCSC.add_miR.160920.refgene.mat \
-ta 0.1 \
-armpeel 1 \
-brlen 0.7 \
-cap 1.5 \
-conf 0.99 \
-td 0.1 \
-genegistic 1 \
-gcm extreme \
-js 4 \
-maxseg 2000 \
-qvt 0.25 \
-rx 0 \
-savegene 1
