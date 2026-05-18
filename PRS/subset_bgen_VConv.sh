#!/bin/bash
#SBATCH --job-name=convBGEN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=30-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=32    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=plink2_%x_%j.log
#SBATCH --mem=512G # RAM per CPU

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0
module load plink/2.00_20211217

plink2 \
    --bgen /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/bgen1p3_cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen ref-first \
    --keep /group/glastonbury/soumick/GWAS/ukbb_genotypes/subIDs_Caucasian.txt \
    --sample /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/bgen1p3_cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.sample \
    --extract /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/cond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.txt \
    --export bgen-1.2 bits=8 \
    --out /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/cond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4 --threads 64

echo "BGEN subsetted, now indexing..."
/ssu/gassu/software/bgen/1.1.7/bgenix -g /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/cond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen -index

#This conversion can also be done using QCTool, but that's super slow.
# module load openblas
# module load gcc

# /ssu/gassu/software/qctool/2.2.1/qctool_v2.2.1 \
#     -filetype bgen \
#     -g /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/bgen1p3_cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen \
#     -og /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/conv_cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen \
#     -ofiletype bgen_v1.2 -bgen-bits 8 -threads 32