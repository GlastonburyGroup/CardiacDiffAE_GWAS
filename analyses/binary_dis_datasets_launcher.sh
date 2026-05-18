#!/bin/bash
#SBATCH --job-name=imatlas
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=12:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=1    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=bindis_%x_%A_%a_%j.log
#SBATCH --mem-per-cpu=22000Mb # RAM per CPU

#SBATCH --array=0-161

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

echo "Command executed: $0 $@"

echo "-----------------------"

module load mpi/2021.2.0

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Check the script!"
    echo ""
}


#function to handle the death of the script!
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

###
###process the arguments

##############################################################################
# Variables & defaults
##############################################################################

# We can still allow command-line overrides using the same pattern:
while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

# Default values for the command-line arguments (if none provided)
# sbatcher="${sbatcher:-/home/soumick.chatterjee/cpurun.sh}"
conda="${conda:-/project/ukbblatent/envs/torchHTBeta2V2UKB}"
tricorder_root="${tricorder_root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder}"
programme="${programme:-analyses/binary_dis_datasets.py}"

processed_phecodes="${processed_phecodes:-/project/ukbblatent/clinicaldata/Atlas/mappedPheX/processed_GP_HI_SR_Phecode_ALL.csv}"
ds_root="${ds_root:-/scratch/glastonbury/datasets/ukbbH5s}"
cov_root="${cov_root:-/group/glastonbury/GWAS/inputs/covariates}"
out_root="${out_root:-/project/ukbblatent/clinicaldata/Atlas/binary_disease_cohorts/MultiOrganV3/ignore_mismatch_inst}"
cov_suffix="${cov_suffix:-woSmoking_woMRICentre}"

##############################################################################
# Set up the arrays of organs, codes, and modes
##############################################################################

# organs=("All_noeye")
organs=("Heart" "Pancreas" "Liver" "Brain" "DXA" "All_noeye")
codes=("Phecode" "PhecodeCh" "PhecodeLvl1")
# modes=("all" "diag" "diag10y") 
modes=("all" "diag" "diag10y" "diag5y" "prog" "prog5y" "near" "near5y" "near2y")

# Build an array of all possible combinations:
combinations=()
for organ in "${organs[@]}"; do
    for code in "${codes[@]}"; do
        for mode in "${modes[@]}"; do
            # Store "organ,code,mode" as a single string (comma-separated or space-separated)
            combinations+=("${organ},${code},${mode}")
        done
    done
done

# Check the array length and compare with $SLURM_ARRAY_TASK_ID
total_combinations=${#combinations[@]}
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Please run via sbatch --array=0-$((total_combinations-1))."
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range (0..$((total_combinations-1)))."
    exit 1
fi

##############################################################################
# Parse the combination for this specific $SLURM_ARRAY_TASK_ID
##############################################################################
IFS=',' read -r organ code mode <<< "${combinations[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "Total combinations = ${total_combinations}"
echo "Selected organ: ${organ}"
echo "Selected code:  ${code}"
echo "Selected mode:  ${mode}"

##############################################################################
# Execute the actual job
##############################################################################

#Change the working directory to the supplied tricorder_root!
cd $tricorder_root

source /home/${USER}/.bashrc
conda activate $conda
echo "Starting the exection of the Python script: $programmePath inside the conda environment $conda, where the python binary is $(which python)";
python $programme --processed_phecodes ${processed_phecodes} --ds_root ${ds_root} --cov_root ${cov_root} --out_root ${out_root} --organs ${organ} --codes ${code} --modes ${mode} --cov_suffix ${cov_suffix} --remove_mismatch_inst