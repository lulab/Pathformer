# Pathformer

[![python >3.6.9](https://img.shields.io/badge/python-3.6.9-brightgreen)](https://www.python.org/) 

## Pathformer: biological pathway informed Transformer model integrating multi-modal data of cancer

Multi-modal biological data integration can provide comprehensive views of gene regulation and cell development. However, conventional integration methods rarely utilize prior biological knowledge and lack interpretability. To address these two challenges, we developed Pathformer, which is to our knowledge the first Transformer-based multi-modal integration model that incorporates biological pathway knowledge. Pathformer leverages criss-cross attention mechanism to capture crosstalk between different biological pathways and between different modalities (i.e., multi-omics). It also utilizes SHapley Additive Explanation method to reveal key pathways, genes, and regulatory mechanisms.Pathformer performs classification tasks with high accuracy and biological interpretability, which can be applied to the liquid biopsy data for noninvasive cancer diagnosis.


![Overview of the Pathformer](method_overview.png)


## Getting Started

To get a local copy up and running, follow these simple steps

### Prerequisites

python 3.6.9, check environments.yml for list of needed packages

### Installation

1.Clone the repo

```git clone https://github.com/lulab/Pathformer.git```

2.Create conda environment

```conda env create --name Pathformer --file=environment.yml```

## Usage

1.Activate the created conda environment

```source activate Pathformer```

2.Data download and preprocessing


