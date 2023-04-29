library(TCGAbiolinks)
library(SummarizedExperiment)
#mRNA
mRNA_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "STAR - Counts")
GDCdownload(mRNA_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
mRNA_data <- GDCprepare(query = mRNA_query,
                   save = TRUE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.mRNA.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.mRNA.RData")
data
expr = assay(data)
expr = as.data.frame(expr)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.mRNA.csv",row.names=TRUE)

#miRNA
miRNA_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Transcriptome Profiling",
                  data.type = "miRNA Expression Quantification",
                  workflow.type = "BCGSC miRNA Profiling")
GDCdownload(miRNA_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
miRNA_data <- GDCprepare(query = miRNA_query,
                   save = TRUE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.miRNA.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.miRNA.RData")
expr = as.data.frame(data)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.miRNA.csv",row.names=TRUE)
#DNA
DNA_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "DNA Methylation",
                  platform = "Illumina Human Methylation 450",
                  legacy = FALSE,
                  data.type ='Methylation Beta Value')
GDCdownload(DNA_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
DNA_data <- GDCprepare(query = DNA_query,
                   save = TRUE,summarizedExperiment=FALSE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.DNAmethy.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.DNAmethy.RData")
expr = as.data.frame(data)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.DNAmethy.csv",row.names=TRUE)

#CNV_seg
CNV_seg_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Copy Number Variation",
                  data.type = "Copy Number Segment")
GDCdownload(CNV_seg_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
CNV_seg_data <- GDCprepare(query = CNV_seg_query,
                   save = TRUE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_seg.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_seg.RData")
expr = as.data.frame(data)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_seg.csv",row.names=TRUE)

#CNV_gene
CNV_gene_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Copy Number Variation",
                  data.type = "Gene Level Copy Number")
query.cnv.cases<-getResults(CNV_gene_query, cols="cases")
length(query.cnv.cases)
query.cnv.cases.dups<-query.cnv.cases[duplicated(query.cnv.cases)]
length(query.cnv.cases.dups)
query.cnv.cases.unique<-unique(query.cnv.cases)
length(query.cnv.cases.unique)
query.cnv.cases.nodups<-setdiff(query.cnv.cases.unique,query.cnv.cases.dups)
length(query.cnv.cases.nodups)

query.cnv.nodups <- GDCquery(project = "TCGA-BRCA", data.category = "Copy Number Variation", data.type = "Gene Level Copy Number",platform="Affymetrix SNP 6.0",legacy=FALSE,barcode=query.cnv.cases.nodups)
GDCdownload(query.cnv.nodups, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")

CNV_gene_data <- GDCprepare(query = query.cnv.nodups,
                   save = TRUE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_gene.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_gene.RData")
data
expr = assay(data)
expr = as.data.frame(expr)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_gene.csv",row.names=TRUE)

#CNV_seg_masker
CNV_seg_query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Copy Number Variation",
                  data.type = "Masked Copy Number Segment")
GDCdownload(CNV_seg_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
CNV_seg_data <- GDCprepare(query = CNV_seg_query,
                   save = TRUE,
                   directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
                   save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_masked_seg.RData")
load("/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_masked_seg.RData")
expr = as.data.frame(data)
write.csv(expr,"/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_masked_seg.csv",row.names=TRUE)

# #CNV_gene_score
# CNV_gene_query <- GDCquery(project = "TCGA-BRCA",
#                   data.category = "Copy Number Variation",
#                   data.type = "Gene Level Copy Number Scores")
# GDCdownload(CNV_gene_query, method = "api",directory="/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data")
# CNV_gene_data <- GDCprepare(query = CNV_gene_query,
#                    save = TRUE,
#                    directory =  "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data",
#                    save.filename = "/apps/home/lulab_liuxiaofan/qhsky/project/multi_omics_paper/data_TCGA/1.raw_data/BRCA.CNV_gene_score.RData")
