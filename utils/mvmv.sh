#!/bin/bash
#SBATCH --job-name=mvUKBRECOH
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=30-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=16    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=sys_%x_%j.log
#SBATCH --mem=256G # RAM per CPU

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

echo "Moving Set0...."
mv /processing_data/shared_datasets/ukbiobank/processed/Set0 /processing_data/shared_datasets/ukbiobank/processed/F20201_Dixon_technique_internal_fat_DICOM_RECOHPipe/
echo "Set0 moved....Moving Set1...."
mv /group/glastonbury/soumick/dataset/ukbbnii/analysis/Set1 /processing_data/shared_datasets/ukbiobank/processed/F20201_Dixon_technique_internal_fat_DICOM_RECOHPipe/
echo "Set1 moved....Moving Set2...."
mv /group/glastonbury/soumick/dataset/ukbbH5s/Set2 /processing_data/shared_datasets/ukbiobank/processed/F20201_Dixon_technique_internal_fat_DICOM_RECOHPipe/
echo "Set2 moved....Done!!!"
