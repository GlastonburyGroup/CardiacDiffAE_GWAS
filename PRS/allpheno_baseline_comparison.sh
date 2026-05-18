#!/bin/bash
#SBATCH --job-name=baselineComp_S28
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
# Per-disease configuration (one entry per SLURM array index)
# ─────────────────────────────────────────────────────────────────────────────
diseases=(
    "atherosclerotic"                # 0 → CVD  (UKB field f.26223)
    "atrial-fibrillation_flutter"    # 1 → AFib (UKB field f.26212)
    "hypertension"                   # 2 → HT   (UKB field f.26244)
    "type-2-diabetes"                # 3 → T2D  (UKB field f.26285)
    "coronary-heart-disease"         # 4 → CAD  (UKB field f.26227)
)

sota_cols=(
    "prs_sota_CVD"
    "prs_sota_AFib"
    "prs_sota_HT"
    "prs_sota_T2D"
    "prs_sota_CAD"
)

sota_labels=(
    "UKB CVD PRS (f.26223)"
    "UKB AFib PRS (f.26212)"
    "UKB Hypertension PRS (f.26244)"
    "UKB T2D PRS (f.26285)"
    "UKB CAD PRS (f.26227)"
)

out_tags=(
    "ukb_CVD_26223"
    "ukb_AFib_26212"
    "ukb_HT_26244"
    "ukb_T2D_26285"
    "ukb_CAD_26227"
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve task-specific variables
# ─────────────────────────────────────────────────────────────────────────────
task_idx=${SLURM_ARRAY_TASK_ID:-0}
dis_name="${diseases[$task_idx]}"
sota_col="${sota_cols[$task_idx]}"
sota_label="${sota_labels[$task_idx]}"
out_tag="${out_tags[$task_idx]}"

echo "Task ${task_idx}: disease=${dis_name}  sota_col=${sota_col}  sota_label=${sota_label}"

# ─────────────────────────────────────────────────────────────────────────────
# General settings (override via --export or sbatch arguments if needed)
# ─────────────────────────────────────────────────────────────────────────────
root="${root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
programme="${programme:-PRS/baseline_comparison_pipeline.py}"
conda="${conda:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

disease_root="${disease_root:-/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2}"

pth_covars="${pth_covars:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv}"

# Pre-processed PRS TSV (bypasses slow per-RDS loading)
pth_prs_processed="${pth_prs_processed:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/all_latents_PRS.tsv}"

pth_prs_sota="${pth_prs_sota:-/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/PRS_82779_MD_12_11_2025_12_17_48.tsv}"

pth_out_root="${pth_out_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/comparisons_rerun/baseline}"

use_variational_inference="${use_variational_inference:-0}"

run_best_single="${run_best_single:-1}"
run_elastic_net="${run_elastic_net:-1}"
run_pca="${run_pca:-1}"
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
    --pth_prs_processed "$pth_prs_processed" \
    --pth_prs_sota "$pth_prs_sota" \
    --pth_dis "$disease_root/${dis_name}.csv" \
    --pth_out_root "$pth_out_root" \
    --pth_out_dir "${out_tag}_V0v2" \
    --sota_col "$sota_col" \
    --sota_label "$sota_label" \
    --orthogonalise \
    $( [[ "$run_best_single" -eq 1 ]]           && echo '--run_best_single'            || echo '--no-run_best_single' ) \
    $( [[ "$run_elastic_net" -eq 1 ]]           && echo '--run_elastic_net'            || echo '--no-run_elastic_net' ) \
    $( [[ "$run_pca" -eq 1 ]]                   && echo '--run_pca'                    || echo '--no-run_pca' ) \
    $( [[ "$run_bayesian_pymc" -eq 1 ]]         && echo '--run_bayesian_pymc'          || echo '--no-run_bayesian_pymc' ) \
    $( [[ "$run_bayesian_bambi" -eq 1 ]]        && echo '--run_bayesian_bambi'         || echo '--no-run_bayesian_bambi' ) \
    $( [[ "$use_variational_inference" -eq 1 ]] && echo '--use_variational_inference'  || echo '--no-use_variational_inference' ) \
    --no-run_with_sota
