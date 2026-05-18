#!/bin/bash

source /home/${USER}/.bashrc
conda activate /scratch/soumick.chatterjee/conda_envs/BeegFSTorchHTBeta2

python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004294 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004295 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0009463 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0000319 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0009506 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0002461 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004260 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004512 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004298 
python /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/utils/get_ols.py --id EFO_0004503 