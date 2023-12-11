# TCGA data preprocessing

TCGA data preprocessing includes multi-modal data download, sample filtering, and gene embedding of multi-modal data. Here we take three classification tasks of breast cancer as an example, including breast cancer early- and late- stage classification, and breast cancer low- and high- survival risk classification.

## 1.TCGA multi-modal data download

We used the “TCGAbiolinks” package of R software to download RNA expression, DNA methylation, DNA CNV and clinical data of breast cancer datasets. Among these downloaded datasets, the RNA expression values were read counts processed by STAR and normalized by TPM, the CpG site levels of DNA methylation data are β-values measured using the Infinium HumanMethylation450 BeadChip, and the DNA CNV data were masked copy number segment and gene level score processed by Gistic2. 

The relevant code can be found in ```/data_preprocessing_TCGA/1.data_download/```.

## 2.TCGA sample filtering

Firstly, we added label information for classification experiments. For breast cancer early- and late- stage classification, we defined stage I and stage II as the early stage and stage III as the late stage according to the "pathologic stage" information in clinical data. For breast cancer low- and high- survival risk classification, we defined samples from patients with survival time greater than 1825 days as low-risk samples and those less than 1825 days as high-risk samples. Finally, we performed further filtering and only retained samples that contained RNA expression, DNA methylation, DNA CNV, and corresponding clinical labels on each cancer dataset.
```bash data_preprocessing_TCGA/2.sample_filtering/log_sample_filter.sh```

The relevant code can be found in ```/data_preprocessing_TCGA/2.sample_filtering/```.

## 3. Gene embedding of multi-modal data

Pathformer supports any number of modalities as input which may have different dimensions, including nucleotide level, fragment level, and gene level. Pathformer’s input for TCGA datasets includes gene-level RNA expression, fragment-level DNA methylation, and both fragment-level and gene-level DNA CNV. To retain the diversity of different modalities, we used a series of statistical indicators to convert different modalities into gene level modal features, and then concatenate these modal features into a compressed multi-modal vector as gene embedding. These statistical indicators include gene level score, count, entropy, minimum, maximum, mean, weighted mean in whole gene, and weighted mean in window.

```bash data_preprocessing_TCGA/3.data_gene_embedding/log_data_gene_embedding.sh```

The relevant code can be found in ```/data_preprocessing_TCGA/3.data_gene_embedding/```. data_all.npy is the gene embedding matrix of each dataset.

```1.1.DNA_methylation_promoterid.py```: Correspond the promoter ID to the gene ID.

```1.2.DNA_methylation.py```: Calculate count of DNA methylation on gene body, maximum of DNA methylation on gene body, minimum of DNA methylation on gene body, and mean of DNA methylation on gene body.

```1.3.DNA_methylation_promoter.py```: Calculate count of DNA methylation on gene promoter, maximum of DNA methylation on gene promoter, minimum of DNA methylation on gene promoter, and mean of DNA methylation on gene promoter.

```2.1.CNV_id.py```: Correspond the CNV ID to the gene ID.

```2.2.CNV_data.py```: Calculate count of DNA CNV,gene level score of DNA CNV.

```2.3.CNV_data_merge.py```: Calculate maximum of DNA CNV, minimum of DNA CNV, and mean of DNA CNV.

```3.1.data_gene_embedding_all.py``` : Organize statistical indicators of multi-modal data for filtered samples.

```3.2.data_gene_embedding_merge.py``` : Merge data into gene embedding data and save it as data_all.npy.
