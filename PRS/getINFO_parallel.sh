#!/bin/bash
#SBATCH --job-name=varinfo_chr%a
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=02-00:00:00    # adjusted walltime as per chromosome processing might take less time
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=16    # number of CPUs per task
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=bgex_chr%a_%x_%j.log
#SBATCH --mem=256G            # RAM per CPU
#SBATCH --array=1-22          # setting up an array job for chromosomes 1-22

exec 2>&1                    # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0
module load openblas
module load gcc/10.2.0

# Adjusting paths for chromosome-wise processing
BGEN_PATH="/processing_data/shared_datasets/ukbiobank/raw_data/genotypes/imputed/ukb22828_c${SLURM_ARRAY_TASK_ID}_b0_v3.bgen"
SAMPLE_PATH="/processing_data/shared_datasets/ukbiobank/raw_data/genotypes/imputed/ukb22828_c${SLURM_ARRAY_TASK_ID}_b0_v3_s487202.sample"
OUT_PATH="/group/glastonbury/soumick/PRS/inputs/common/bgex_chrs/ukb22828_c${SLURM_ARRAY_TASK_ID}_b0_v3_variant_info.txt"

/group/glastonbury/soumick/bin/bgex/bgex \
    --bgens ${BGEN_PATH} \
    --samples ${SAMPLE_PATH} \
    --info \
    --min-info 0.4 \
    --out ${OUT_PATH}
