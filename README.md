# Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20281276.svg)](https://doi.org/10.5281/zenodo.20281276)
[![Preprint](https://img.shields.io/badge/medRxiv-2024.11.04.24316700-blue)](https://doi.org/10.1101/2024.11.04.24316700)
[![Project page](https://img.shields.io/badge/project-page-informational)](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Official code for the paper *"Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture"*, accepted for publication in **Nature Communications**.

**Authors:** Sara Ometto\*, Soumick Chatterjee\*, Andrea Mario Vergani, Arianna Landini, Sodbo Sharapov, Edoardo Giacopuzzi, Alessia Visconti, Emanuele Bianchi, Federica Santonastaso, Emanuel M. Soda, Francesco Cisternino, Carlo Andrea Pivato, Francesca Ieva, Emanuele Di Angelantonio, Nicola Pirastu, Craig A. Glastonbury.
\* Joint first authors.

An earlier version is available as a preprint on medRxiv: <https://doi.org/10.1101/2024.11.04.24316700>. The project page, with links to data, figures, and related resources, lives at <https://glastonburygroup.github.io/CardiacDiffAE_GWAS/>.

## What this repository contains

This repository holds the analysis code accompanying the paper, with one exception: the deep learning pipeline itself is maintained separately, at [GlastonburyGroup/ImLatent](https://github.com/GlastonburyGroup/ImLatent). The scripts here cover:

- **`preprocess/`** — preparation of the UK Biobank cardiac MRI data and construction of the H5 datasets used downstream.
- **`H5tools/`** — utilities for working with those H5 datasets (cropping, masking, conversion to Zarr/LMDB, traversal).
- **`postprocess/`** — handling of the latent embeddings produced by the diffusion autoencoder, including PCA, pairwise CCA, and multi-view CCA-based merging.
- **`GWAS/`** — scripts to configure and launch the GWAS, generate covariates, run conditional analyses, merge per-chromosome results, and produce Miami and QQ plots.
- **`PRS/`** — polygenic risk score modelling across phenotypes and disease endpoints, including pan-cohort evaluation.
- **`analyses/`** — downstream analyses, including binary disease prediction and the machine learning experiments reported in the paper.
- **`exome/`** — post-processing of exome-wide association results.
- **`UKBB/`** and **`DNANexus_UKBRAP/`** — helpers for working with the UK Biobank and the DNAnexus UKB-RAP platform.
- **`utils/`** — shared utilities.

For the deep learning component, please see the [ImLatent](https://github.com/GlastonburyGroup/ImLatent) repository.

## Getting started

Dependencies are managed with [Poetry](https://python-poetry.org/). Once Poetry is installed, any script can be run from the repository root without setting anything else up manually:

```bash
poetry run python -m preprocess.createH5s.createH5_MR_DICOM
```

If you would rather not prefix every command, activate the environment using the [Poetry shell plugin](https://github.com/python-poetry/poetry-plugin-shell):

```bash
poetry shell
```

A `Dockerfile` is also provided for reproducible execution.

## Documentation

Step-by-step notes for the more involved parts of the pipeline live in [`instructions/`](instructions/), including how to build the cardiac MRI H5 dataset and how to prepare UK Biobank field 20208.

## Citation

If this code or any part of the analysis is useful in your work, please cite the paper. The Nature Communications reference will be added once it is in print; in the meantime, the preprint can be cited as:

```bibtex
@article{Ometto2024.11.04.24316700,
  author    = {Ometto, Sara and Chatterjee, Soumick and Vergani, Andrea Mario and Landini, Arianna and Sharapov, Sodbo and Giacopuzzi, Edoardo and Visconti, Alessia and Bianchi, Emanuele and Santonastaso, Federica and Soda, Emanuel M and Cisternino, Francesco and Pivato, Carlo Andrea and Ieva, Francesca and Di Angelantonio, Emanuele and Pirastu, Nicola and Glastonbury, Craig A},
  title     = {Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture},
  year      = {2024},
  doi       = {10.1101/2024.11.04.24316700},
  publisher = {Cold Spring Harbor Laboratory Press},
  url       = {https://www.medrxiv.org/content/10.1101/2024.11.04.24316700},
  journal   = {medRxiv},
  note      = {Ometto and Chatterjee contributed equally. Accepted at Nature Communications.}
}
```

The snapshot of the codebase at the point of acceptance is archived on Zenodo:

```bibtex
@software{cardiacdiffae_gwas_zenodo,
  title     = {GlastonburyGroup/CardiacDiffAE\_GWAS: Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20281276},
  url       = {https://doi.org/10.5281/zenodo.20281276}
}
```

## Contact

For questions about the code or the analyses, please open an issue or get in touch with Soumick Chatterjee (<soumick.chatterjee@fht.org>, <contact@soumick.com>).
