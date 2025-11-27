# Data and Scripts - Viral Mimicry Analysis

**Manuscript:** Structural-functional trade-off constrains viral protein mimicry of human immune pathways

**Authors:** [Your names]

**Date:** 2025

---

## Overview

This directory contains all data and Python scripts required to reproduce the analyses reported in the manuscript. Scripts are numbered sequentially and should be executed in order.

---

## Directory Structure

```
data_and_scripts/
├── README.md
├── requirements.txt
├── virus_to_human_top5_neighbors_final_similarity.csv  # Main dataset
├── sequences/                                           # Protein sequences
│   ├── human/
│   │   └── human_proteome.fasta                        # Homo sapiens proteome
│   └── viruses/                                        # Viral proteomes by taxonomy
│       ├── Picornaviridae/
│       ├── Herpesviridae/
│       ├── Retroviridae/
│       └── [other families]/
└── script_*.py                                          # Analysis scripts
```

---

## Requirements

### Software Dependencies

- Python >= 3.8
- Required packages (install via `requirements.txt`):
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - scipy >= 1.7.0
  - matplotlib >= 3.4.0
  - seaborn >= 0.11.0
  - scikit-learn >= 1.0.0
  - gseapy >= 1.0.0

### Installation

```bash
pip install -r requirements.txt
```

---

## Data Files

### 1. Main Dataset

**File:** `virus_to_human_top5_neighbors_final_similarity.csv`

**Description:** Pre-computed similarity scores between viral and human proteins

**Contents:**

- 3,945 virus-human protein pairs
- 1,056 unique viral proteins
- 691 unique human proteins
- 50 human-infecting viruses
- 22 ICTV viral families

**Columns:**

- `viral_protein_id`: Viral protein identifier
- `human_protein_id`: Human protein identifier
- `S_sim`: Structural similarity (ESM-2 embeddings)
- `F_sim`: Functional similarity (UniProt annotations)
- `VMS`: Virus Mimicry Score = 0.5 × S_sim + 0.5 × F_sim

### 2. Protein Sequences

**Directory:** `sequences/`

**Human proteome:** `sequences/human/human_proteome.fasta`

- Source: UniProt (UP000005640)
- Reviewed entries only

**Viral proteomes:** `sequences/viruses/[Family]/`

- Source: UniProt
- Organized by ICTV taxonomy
- 50 human-infecting viruses
- 22 viral families

---

## Analysis Scripts

Scripts are numbered to indicate execution order:

| Script                                        | Analysis Type | Description                              |
| --------------------------------------------- | ------------- | ---------------------------------------- |
| `script_01_exploratory_analysis.py`         | Descriptive   | Dataset statistics and distributions     |
| `script_02_distributions.py`                | Visualization | Distribution plots for S_sim, F_sim, VMS |
| `script_03_normality_tests.py`              | Statistical   | Normality testing (Shapiro-Wilk, K-S)    |
| `script_04_global_correlation.py`           | Correlation   | Global S_sim vs F_sim correlation        |
| `script_05_stratified_correlation.py`       | Correlation   | VMS-stratified correlation analysis      |
| `script_06_benchmarking_internal.py`        | Validation    | Dataset quality benchmarking             |
| `script_07_permutation_test.py`             | Validation    | Permutation test (n=1000)                |
| `script_08_predictive_analysis.py`          | Modeling      | Predictive regression models             |
| `script_09_bar_visualization.py`            | Visualization | High-mimicry pair plots                  |
| `script_11_normalized_family_comparison.py` | Comparative   | Family-level normalized analysis         |
| `script_KEGG_pathway_enrichment.py`         | Enrichment    | KEGG pathway enrichment analysis         |
| `script_NN_baseline_final.py`               | Modeling      | Neural network VMS prediction            |

---

## Execution

### Running All Analyses

Execute scripts sequentially:

```bash
python3 script_01_exploratory_analysis.py
python3 script_02_distributions.py
python3 script_03_normality_tests.py
python3 script_04_global_correlation.py
python3 script_05_stratified_correlation.py
python3 script_06_benchmarking_internal.py
python3 script_07_permutation_test.py
python3 script_08_predictive_analysis.py
python3 script_09_bar_visualization.py
python3 script_11_normalized_family_comparison.py
python3 script_KEGG_pathway_enrichment.py
python3 script_NN_baseline_final.py
```

### Running Individual Analyses

Each script can be executed independently. Ensure the main dataset file is in the same directory.

---

## Output Files

Scripts generate the following types of output:

- **Statistical results:** `results_*.csv`
- **Analysis reports:** `report_*.md`
- **Figures:** `fig_*.png` (600 DPI, publication-ready)
- **Models:** `model_*.pkl` (trained ML models)

All outputs are saved in the working directory.

---

## Reproducibility

### Random Seed

All stochastic procedures use fixed random seed (42) to ensure reproducibility:

- Permutation tests
- Neural network training
- Train-test splits

### Statistical Methods

- Non-parametric tests (Spearman correlation, Kruskal-Wallis)
- Permutation-based validation
- Multiple testing correction: Benjamini-Hochberg FDR

### Figure Settings

- Resolution: 600 DPI
- Format: PNG
- Font sizes: Publication-ready

---

## Key Analyses

1. **Global correlation analysis** (script_04): Tests overall relationship between structural and functional similarity
2. **Stratified correlation analysis** (script_05): Reveals trade-off within VMS bins
3. **Permutation test** (script_07): Validates statistical significance of observed patterns
4. **KEGG enrichment** (script_KEGG): Identifies enriched immune pathways
5. **Neural network validation** (script_NN_baseline_final): Demonstrates predictive power of features

---

## Citation

If you use these scripts, please cite:

[Your manuscript citation here]

---

## Contact

For questions or issues, please contact: **beltran.lissabet.jf@gmail.com**

---

## License

### Code (Python Scripts)

**MIT License**

Copyright (c) 2025 Jorge F. Beltrán

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Data and Results

**Creative Commons Attribution 4.0 International (CC-BY 4.0)**

The datasets and results in this repository are licensed under a Creative Commons Attribution 4.0 International License.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

To view a copy of this license, visit: https://creativecommons.org/licenses/by/4.0/

---

**Last updated:** 2025-11-20
