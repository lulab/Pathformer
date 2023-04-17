# Pathformer

### Pathformer: biological pathway informed Transformer model integrating multi-modal data of cancer

Multi-modal biological data integration can provide comprehensive views of gene regulation and cell development. However, conventional integration methods rarely utilize prior biological knowledge and lack interpretability. To address these two challenges, we developed Pathformer, which is to our knowledge the first Transformer-based multi-modal integration model that incorporates biological pathway knowledge. Pathformer leverages criss-cross attention mechanism to capture crosstalk between different biological pathways and between different modalities (i.e., multi-omics). It also utilizes SHapley Additive Explanation method to reveal key pathways, genes, and regulatory mechanisms.Pathformer performs classification tasks with high accuracy and biological interpretability, which can be applied to the liquid biopsy data for noninvasive cancer diagnosis.


![Overview of the Pathformer](method_overview.png)

