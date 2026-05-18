#!/bin/bash
#SBATCH --job-name=bgen2pgen
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
    --bgen /scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.bgen ref-first \
    --sample /scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.sample \
    --make-pgen --out /group/glastonbury/soumick/GWAS/ukbb_genotypes/imputed_pgen/step2_dataset_autosomes.mac100  --threads 32