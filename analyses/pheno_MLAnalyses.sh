#!/bin/bash
#SBATCH --job-name=mldisPan259
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=10-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=1    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=phenoML_%x_%A_%a_%j.log
#SBATCH --mem-per-cpu=15000Mb # RAM per CPU

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

# Populate the associative array with keys and values
declare -A organ_resroots
organ_resroots["Heart"]="/project/ukbblatent/Out/Gen2/HeartMRI/Results"
organ_resroots["Brain"]="/project/ukbblatent/BrainMRI/Models/Results"
organ_resroots["Pancreas"]="/project/ukbblatent/Out/Gen2/PancreasMRI/Results"
organ_resroots["Liver"]="/project/ukbblatent/Out/Gen2/LiverMRI/Results"
# organ_resroots["DXA"]="/project/ukbblatent/Out/Gen2/DXA/Results"
# organ_resroots["DIXON"]="/project/ukbblatent/Out/Gen2/DIXON/Results"


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

# Declare an associative array
declare -A paths

# Populate the associative array with compound keys
paths["Heart_Main"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/cardiacMRI"
paths["Heart_PWA"]="/project/ukbblatent/clinicaldata/Atlas/Phenos"
paths["Brain_Main"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/brainMRI"
paths["Pancreas_Abdominal"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/abdominalMRI"
paths["Liver_Abdominal"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/abdominalMRI"
paths["DXA_Main"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/DXA"
paths["DIXON_Abdominal"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/abdominalMRI"
paths["DIXON_DXA"]="/project/ukbblatent/clinicaldata/Atlas/IDPs/DXA"

# Function to get the path based on organ and type
get_path() {
    local organ="$1"
    local type="$2"
    local key="${organ}_${type}"
    
    if [[ -v paths["$key"] ]]; then
        echo "${paths["$key"]}"
    else
        echo "Path not found for Organ: $organ, Type: $type"
    fi
}
# Default values for the command-line arguments (if none provided)
conda="${conda:-}"
tricorder_root="${tricorder_root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder}"
programme="${programme:-analyses/latents/analysers/ml_analyses.py}"
consolidate_programme="${consolidate_programme:-analyses/latents/analysers/pheno_mlanalyses_summeriser.py}"

cov_root="${cov_root:-/group/glastonbury/GWAS/inputs/covariates}"
out_root="${out_root:-/project/ukbblatent/soumick/ML_analyses/Atlas/MultiOrganV3_Initial}"
cov_suffix="${cov_suffix:-woSmoking_woMRICentre}"

n_jobs="${n_jobs:-6}"
max_iter_CV="${max_iter_CV:-100}"
max_iter="${max_iter:-100}"
l1_penalty="${l1_penalty:-0.05}"

specific_phenogroup="${specific_phenogroup:-}"

#need to update for each model run
res_root="${res_root:-}"
trainID="${trainID:-}"
dsV="${dsV:-}" #need to supply this when the training ID does not have the version number or the correct version number
organ="${organ:-}" #need to supply this when the training ID does not have the organ or the correct organ (including the case)
run_tag="${run_tag:-}"
outdir="${outdir:-}"

##############################################################################
# Prepare the input arguments for the Python script
##############################################################################
if [ -z "$organ" ]; then #if organ is supplied, that will be used.
    organ="${trainID#*_}"  # Remove everything before the first underscore
    organ="${organ%%_*}"  # Remove everything after the second underscore

    if [[ $organ == Brain* ]]; then
        organ="Brain"
    fi

    if [ -z "$res_root" ]; then
        echo "res_root is not set. Using the default one for $organ."
        res_root="${organ_resroots[$organ]}"
    fi
fi

echo "Starting analysis for organ: $organ with trainID: $trainID"

embH5="${res_root}/${trainID}/${outdir}/emb.h5"

ds="${trainID%%_*}" # Extract everything before the first '_' from the trainID
fieldID="${ds%v*}"  # Extract everything before 'v'
if [ -z "$dsV" ]; then #if dsV is supplied, that will be used.
    dsV="${ds#*v}"  # Extract everything after 'v'
fi

cov_dir=$(find "$cov_root" -type d -name "${fieldID}*${dsV}" -print | head -n 1)
if [[ -n "$cov_dir" ]]; then
    cov_path=$(find "$cov_dir" -type f \( -name "cov_*${cov_suffix}.tsv" -o -name "cov_*${cov_suffix}.csv" \) -print | head -n 1)
    if [[ -z "$cov_path" ]]; then
        echo "No matching .tsv or .csv file found in $cov_dir that ends with $cov_suffix."
        exit 1
    fi
else
    echo "${cov_dir} folder not found."
    exit 1
fi

model="${trainID##*_}"
model_tag="${model}_${ds}_${organ}_${run_tag}"

##############################################################################
# Execute the actual job
##############################################################################

#Change the working directory to the supplied tricorder_root!
cd $tricorder_root

source /home/${USER}/.bashrc
if [ -n "$conda" ]; then
    echo "Starting the execution of the Python script: $programmePath inside the conda environment $conda";   
    conda activate $conda
    # python $programmePath $args
else
    echo "Starting the execution of the Python script: $programmePath using Poetry";
    # poetry run python $programmePath $args
fi

# Loop through all types for the specified organ
if [[ -z "$specific_phenogroup" ]]; then    
    echo "Fetching all types for organ: $organ"
    for key in "${!paths[@]}"; do
        if [[ "$key" == "${organ}_"* ]]; then
            type="${key#*_}"
            path_tsvs=$(get_path "$organ" "$type")
            if [[ "$type" == "Main" ]]; then
                type="IDP"
            else
                type="Pheno${type}"
            fi
            out_path="${out_root}/${organ}/${type}"
            python $programme --embH5 ${embH5} --model_tag ${model_tag} --out_path ${out_path} --path_tsvs ${path_tsvs} --cov_path ${cov_path} --n_jobs ${n_jobs} --max_iter_CV ${max_iter_CV} --max_iter ${max_iter} --l1_penalty ${l1_penalty} --no-is_disease --index_col "f.eid"
        fi
    done
else
    # Fetch the path for the specified organ and type
    echo "Fetching path for organ: $organ, type: $specific_phenogroup"
    path_tsvs=$(get_path "$organ" "$specific_phenogroup")
    if [[ "$specific_phenogroup" == "Main" ]]; then
        type="IDP"
    else
        type="Pheno${specific_phenogroup}"
    fi
    out_path="${out_root}/${organ}/${type}"
    poetry run python $programme --embH5 ${embH5} --model_tag ${model_tag} --out_path ${out_path} --path_tsvs ${path_tsvs} --cov_path ${cov_path} --n_jobs ${n_jobs} --max_iter_CV ${max_iter_CV} --max_iter ${max_iter} --l1_penalty ${l1_penalty} --no-is_disease --index_col "f.eid"
fi

#consolidate the results
echo "Consolidating the results for the organ: $organ"
poetry run python $consolidate_programme --out_root ${out_root} --organ ${organ} --model_tag ${model_tag}