# Unsupervised cardiac MRI phenotyping with 3D diffusion autoencoders reveals novel genetic insights

The official code for the paper, "Unsupervised Cardiac MRI Phenotyping with 3D Diffusion Autoencoders Reveals Novel Genetic Insights". This repository contains all scripts used in this project, except for the deep learning pipeline (DL-pipeline). It includes scripts for pre-processing raw datasets from the UK Biobank, post-processing and analysing the latent embeddings, as well as conducting downstream analyses. The deep learning pipeline for unsupervised latent phenotyping using the 3D diffusion autoencoder can be found here: https://github.com/GlastonburyGroup/ImLatent.

This repository is organised into multiple folders, each containing standalone scripts for specific purposes. Additional scripts (e.g., for launching GWAS, performing post-GWAS analyses, PRS modelling, etc.) will be uploaded in the near future, along with detailed documentation. In the meantime, contact us for any questions or requests (Email: soumick.chatterjee@fht.org)

## Citation
If you find this work useful or utilise any code from this repository in your research, please consider citing us:
```bibtex
@article{Ometto2024.11.04.24316700,
            author       = {Ometto, Sara and Chatterjee, Soumick and Vergani, Andrea Mario and Landini, Arianna and Sharapov, Sodbo and Giacopuzzi, Edoardo and Visconti, Alessia and Bianchi, Emanuele and Santonastaso, Federica and Soda, Emanuel M and Cisternino, Francesco and Ieva, Francesca and Di Angelantonio, Emanuele and Pirastu, Nicola and Glastonbury, Craig A},
            title        = {Unsupervised cardiac MRI phenotyping with 3D diffusion autoencoders reveals novel genetic insights},
            elocation-id = {2024.11.04.24316700},
            year         = {2024},
            doi          = {10.1101/2024.11.04.24316700},
            publisher    = {Cold Spring Harbor Laboratory Press},
            url          = {https://www.medrxiv.org/content/early/2024/11/05/2024.11.04.24316700},
            journal      = {medRxiv}
          }  
```