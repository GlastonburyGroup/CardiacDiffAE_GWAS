#!/bin/bash
#SBATCH --job-name=mPRSV2ppr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=1-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=4    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=prs_%x_%j.log
#SBATCH --mem-per-cpu=9000Mb # RAM per CPU
#SBATCH --array=0-18

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Logs a SLURM job with a GPU Node."
    echo ""
    echo "usage: $programmename --root string --programme string --conda string --args \"string\" or \" \""
    echo ""
    echo "  --root    string           The program root, with the trailing slash (Default: /group/glastonbury/soumick/MyCodes/)"
    echo "                             [If no root is to be supplied, then put quotes with a space in between as its contents]"
    echo "  --programme string         The python file to run (Default: main.py)"
    echo "  --args    string           List of command line arguments, supplied as a single string within quotes (Default: Nothing/Blank)"
    echo "                             [Example: \"--modelID 2 --dataset UKB\". These will be supplied as arguments to the programme]"
    echo "  --conda   string           Name of the Conda env (Default: torchHTBeta2)"
    echo ""
}


#function to handle the death of the script!
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

###
###process the arguments

#read the different keyworded commandline arguments
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

#read config file for the supplied arguments
if [ -n "$config_file" ]; then
  source "$config_file"
else
  echo "Error: Configuration file not found."
  exit 1
fi

#set the default values for the commandline arguments

#general arguments
root="${root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
programme="${programme:-PRS/multiPRS_predictPanCohort_V2.py}"
conda="${conda:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

#arguments for the programme
prs_res_root="${prs_res_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc}"
rds_pres_prefix="${rds_pres_prefix:-run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_}"
rds_pres_suffix="${rds_pres_suffix:-.fullDS.auto.mod.LDPred2.rds}" #for inf model: .fullDS.inf.mod.LDPred2.rds , for auto model: .fullDS.auto.mod.LDPred2.rds , for grid model: .fullDS.grid.mod.LDPred2.rds
rds_tag_prs="${rds_tag_prs:-auto.mod}" #for inf model: inf.mod , for auto model: auto.mod , for grid model: grid.mod
tag_data="${tag_data:-resNdata.basic}"
tag_prs="${tag_prs:-pred_auto}" #for inf model: pred_inf , for auto model: pred_auto , for grid model: pred_grid

disease_root="${disease_root:-/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0}"
col_disease="${col_disease:-BinCAT_Disease}"
min_sub="${min_sub:-1000}"

output_root="${output_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/newcovsets_V0v2/4paper_caucasian_king0p0625_grouped}"
out_tag="${out_tag:-panCohortV2_auto_lw_gw_10kIT_kingB4ldpred2}"

ext_covar="${ext_covar:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv}"
cont_covar_cols="${cont_covar_cols:-Age,Sex,BMI}"
nPCs_covar="${nPCs_covar:-0}"
cat_covar_cols="${cat_covar_cols:-CAT_Smoking}"
lassoCV_max_iter="${lassoCV_max_iter:-10000}"
mode_multi="${mode_multi:-Lasso}"
threads="${threads:-6}"

do_singlePRS="${do_singlePRS:-1}" # Run single PRS models
do_singlePRSCovar="${do_singlePRSCovar:-1}" # Run single PRS + Covariate models
do_singlePRSCovarNorm="${do_singlePRSCovarNorm:-0}" # Run single PRS + Covariate models, with normalisation
do_covar="${do_covar:-1}" # Run Covariate models
do_covarNorm="${do_covarNorm:-0}" # Run Covariate models, with normalisation
do_nonPCCovar="${do_nonPCCovar:-0}" # Additionally run models without PC covariates

do_multiPRS="${do_multiPRS:-1}" # Run multiPRS models
do_multiPRSNorm="${do_multiPRSNorm:-0}" # Run multi normalised PRS models
do_multiPRSCovar="${do_multiPRSCovar:-1}" # Run multiPRS + Covariate models
do_multiPRSNormCovar="${do_multiPRSNormCovar:-0}" # Run multi normalised PRS + Covariate models

###
###Start of the actual script, after reading all the arguments

#Setup conda
source /home/${USER}/.bashrc
conda activate $conda

disease_csvs=("$disease_root"/*.csv)
disease_csv=${disease_csvs[$SLURM_ARRAY_TASK_ID]}
disease=$(basename $disease_csv .csv)
echo "Processing Disease: $disease"
echo "Number of PCs in use: $nPCs_covar"

cd $root

python $programme \
    --prs_res_root "$prs_res_root" \
    --rds_pres_prefix "$rds_pres_prefix" \
    --rds_pres_suffix "$rds_pres_suffix" \
    --rds_tag_prs "$rds_tag_prs" \
    --tag_data "$tag_data" \
    --tag_prs "$tag_prs" \
    --disease_csv "$disease_csv" \
    --col_disease "$col_disease" \
    --min_sub "$min_sub" \
    --output_root "$output_root/$out_tag" \
    --ext_covar "$ext_covar" \
    --cont_covar_cols "$cont_covar_cols" \
    --nPCs_covar "$nPCs_covar" \
    --cat_covar_cols "$cat_covar_cols" \
    --lassoCV_max_iter "$lassoCV_max_iter" \
    --mode_multi "$mode_multi" \
    --threads "$threads" \
    $( [[ "$do_singlePRS" -eq 1 ]] && echo '--do_singlePRS' || echo '--no-do_singlePRS' ) \
    $( [[ "$do_singlePRSCovar" -eq 1 ]] && echo '--do_singlePRSCovar' || echo '--no-do_singlePRSCovar' ) \
    $( [[ "$do_singlePRSCovarNorm" -eq 1 ]] && echo '--do_singlePRSCovarNorm' || echo '--no-do_singlePRSCovarNorm' ) \
    $( [[ "$do_covar" -eq 1 ]] && echo '--do_covar' || echo '--no-do_covar' ) \
    $( [[ "$do_covarNorm" -eq 1 ]] && echo '--do_covarNorm' || echo '--no-do_covarNorm' ) \
    $( [[ "$do_nonPCCovar" -eq 1 ]] && echo '--do_nonPCCovar' || echo '--no-do_nonPCCovar' ) \
    $( [[ "$do_multiPRS" -eq 1 ]] && echo '--do_multiPRS' || echo '--no-do_multiPRS' ) \
    $( [[ "$do_multiPRSNorm" -eq 1 ]] && echo '--do_multiPRSNorm' || echo '--no-do_multiPRSNorm' ) \
    $( [[ "$do_multiPRSCovar" -eq 1 ]] && echo '--do_multiPRSCovar' || echo '--no-do_multiPRSCovar' ) \
    $( [[ "$do_multiPRSNormCovar" -eq 1 ]] && echo '--do_multiPRSNormCovar' || echo '--no-do_multiPRSNormCovar' )