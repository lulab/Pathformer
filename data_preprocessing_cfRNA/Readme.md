# Liquid biopsy data preprocessing

To further verify the effect of Pathformer in cancer diagnosis, we collected three types of body fluid datasets: the plasma dataset, the extracellular vesicle (EV) dataset, and the platelet dataset. Liquid biopsie data preprocessing includes bioinformatics pipeline for multi-modal data and gene embedding of multi-modal data.

## 1.Bioinformatics pipeline for multi-modal liquid biopsie data

For body fluid datasets, we used seven modalities at the RNA level as Pathformer’s input, including RNA expression, RNA splicing, RNA editing, RNA alternative promoter (RNA alt. promoter), RNA allele-specific expression (RNA ASE), RNA single nucleotide variations (RNA SNV), and chimeric RNA. 

We used a bioinformatics pipeline to preprocess raw sequence reads into datasets of different modalities. Firstly, we used cutadapt tool to trim adaptors and low-quality reads, and then removed the reads which can be mapped to ERCC’s spike-in sequences, NCBI’s UniVec sequences (vector contamination), and human rRNA sequences by STAR software. Next, we applied STAR software to map all the retained unmapped reads to the hg38 genome index built with the GENCODE v27 annotation and calculated seven modalities at the RNA level based on the mapping result. The details of the calculation processes are as follows: (1) RNA expression data were read counts aggerated to gene by featureCounts and were normalized by TPM. (2) RNA alternative promoter data represented transcript isoform abundances quantified by salmon and were normalized by TPM. We only selected isoforms with transcription start sites within 10 bp (sharing the same promoter) and TPMs greater than 1. (3) RNA splicing data were a series of alternative splicing events with the percent spliced-in (PSI) score calculated using rMATs-turbo. (4) As for RNA editing data, editing sites were identified by GATK ASEReadCounter based on REDIportal and editing ratios of editing sites were defined as allele count divided by total count. (5) In RNA allele-specific expression data, allele-specific expression gene site were identified by GATK ASEReadCounter based on SNP sites and allelic expressions (AE, AE = |0.5 − Reference ratio |, Reference ratio = Reference reads/Total reads) were calculated for all sites with ≥16 reads.  (6) In RNA single nucleotide variations data, GATK SplitNCigarReads was used to split intron-spanning reads for confident SNP calling at RNA level. GATK HaplotypeCaller and GATK VariantFilteration were used to identify and filter alterations. Allele fraction was defined as allele count divided by total count (reference count and allele count).  (7) Chimeric RNA data were identified by remapping unaligned reads to chimeric junctions by STAR-fusion. Chimera references were based on GTex and ChimerDB-v3.

Bioinformatics pipeline of multi-modal liquid biopsie data is developed by our laboratory, whic can be seen in https://github.com/tyh-19/Pipeline-for-multiomics. If you have any questions, you can contact tyh19@mails.tsinghua.edu.cn.


## 2. Gene embedding of multi-modal liquid biopsie data

Pathformer supports any number of modalities as input which may have different dimensions, including nucleotide level, fragment level, and gene level. Pathformer’s input for liquid biopsy datasets includes gene-level RNA expression; fragment-level RNA alternative promoter, RNA splicing, and chimeric RNA; and nucleotide-level RNA editing, RNA ASE, and RNA SNV. To address this, we used multiple statistical indicators as gene embeddings to retain the gene diversity across different modalities. These statistical indicators include gene level score, count, entropy, minimum, maximum, mean, weighted mean in whole gene, and weighted mean in window. 

```bash data_preprocessing_cfRNA/1.data_gene_embedding/log_data_gene_embedding.sh```

The relevant code can be found in ```/data_preprocessing_cfRNA/1.data_gene_embedding/```. data_all.npy is the gene embedding matrix of each dataset.

