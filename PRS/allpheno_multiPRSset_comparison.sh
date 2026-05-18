#!/bin/bash
#SBATCH --job-name=multiSetComp_S29
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=prs_%x_%A_%a.log
#SBATCH --mem=2G
#SBATCH --array=0-4

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID

module load mpi/2021.2.0

# ─────────────────────────────────────────────────────────────────────────────
# Per-disease configuration (one entry per SLURM array index).
# ─────────────────────────────────────────────────────────────────────────────
diseases=(
    "atherosclerotic"                # 0 → CVD
    "atrial-fibrillation_flutter"    # 1 → AFib
    "hypertension"                   # 2 → HT
    "type-2-diabetes"                # 3 → T2D
    "coronary-heart-disease"         # 4 → CAD
)

out_tags=(
    "CVD"
    "AFib"
    "HT"
    "T2D"
    "CAD"
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve task-specific variables
# ─────────────────────────────────────────────────────────────────────────────
task_idx=${SLURM_ARRAY_TASK_ID:-0}
dis_name="${diseases[$task_idx]}"
out_tag="${out_tags[$task_idx]}"

echo "Task ${task_idx}: disease=${dis_name}  out_tag=${out_tag}"

# ─────────────────────────────────────────────────────────────────────────────
# General settings
# ─────────────────────────────────────────────────────────────────────────────
root="${root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
programme="${programme:-PRS/multi_prs_set_comparison_pipeline.py}"
conda="${conda:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

disease_root="${disease_root:-/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2}"

pth_covars="${pth_covars:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv}"

# PRS set 1: DiffAE Latent PRS (reference set)
prs_set1_name="${prs_set1_name:-Latent multi-PRS}"
prs_set1_root="${prs_set1_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc}"
prs_set1_prefix="${prs_set1_prefix:-run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_}"
prs_set1_suffix="${prs_set1_suffix:-.fullDS.auto.mod.LDPred2.rds}"

# PRS set 2: Cardiac IDP PRS
prs_set2_name="${prs_set2_name:-Cardiac IDPs}"
prs_set2_root="${prs_set2_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/IDPs/cardiac}"
prs_set2_prefix="${prs_set2_prefix:-run_ext_basic_DiffAEking0p0625_lw_gw_indep_FiltMAF_}"
prs_set2_suffix="${prs_set2_suffix:-.fullDS.auto.mod.LDPred2.rds}"

pth_out_root="${pth_out_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/comparisons_rerun/multi_set}"

use_variational_inference="${use_variational_inference:-0}"

orthogonalise="${orthogonalise:-0}"
use_covariates_as_inputs="${use_covariates_as_inputs:-1}"
run_best_single="${run_best_single:-1}"
run_elastic_net="${run_elastic_net:-1}"
run_pca="${run_pca:-0}"
run_bayesian_pymc="${run_bayesian_pymc:-0}"
run_bayesian_bambi="${run_bayesian_bambi:-0}"

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
source /home/${USER}/.bashrc
conda activate $conda

cd $root
$conda/bin/python $programme \
    --pth_covars "$pth_covars" \
    --pth_dis "$disease_root/${dis_name}.csv" \
    --pth_out_root "$pth_out_root" \
    --pth_out_dir "${out_tag}_LatentsVsCardiacIDP_V0v2" \
    --prs_set_name "$prs_set1_name" \
    --prs_set_root "$prs_set1_root" \
    --rds_pres_prefix "$prs_set1_prefix" \
    --rds_pres_suffix "$prs_set1_suffix" \
    --prs_set_name "$prs_set2_name" \
    --prs_set_root "$prs_set2_root" \
    --rds_pres_prefix "$prs_set2_prefix" \
    --rds_pres_suffix "$prs_set2_suffix" \
    --reference_set "$prs_set1_name" \
    --orthogonalise \
    $( [[ "$run_best_single" -eq 1 ]]           && echo '--run_best_single'            || echo '--no-run_best_single' ) \
    $( [[ "$run_elastic_net" -eq 1 ]]           && echo '--run_elastic_net'            || echo '--no-run_elastic_net' ) \
    $( [[ "$run_pca" -eq 1 ]]                   && echo '--run_pca'                    || echo '--no-run_pca' ) \
    $( [[ "$run_bayesian_pymc" -eq 1 ]]         && echo '--run_bayesian_pymc'          || echo '--no-run_bayesian_pymc' ) \
    $( [[ "$run_bayesian_bambi" -eq 1 ]]        && echo '--run_bayesian_bambi'         || echo '--no-run_bayesian_bambi' ) \
    $( [[ "$use_variational_inference" -eq 1 ]] && echo '--use_variational_inference'  || echo '--no-use_variational_inference' ) \
    $( [[ "$use_covariates_as_inputs" -eq 1 ]]  && echo '--use_covariates_as_inputs'   || echo '--no-use_covariates_as_inputs' )
