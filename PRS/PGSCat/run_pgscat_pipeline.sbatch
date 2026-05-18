#!/bin/bash
#SBATCH --job-name=pgscat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=pgscat_%x_%j.log

# ===================================================================
# PGS Catalog publication pipeline (single entry point)
# -------------------------------------------------------------------
# Produces, for the F20208v3_DiffAE 49 selected latents:
#   1. Per-latent PGS Catalog v2.0 scoring files (gzipped .txt.gz),
#      using ONLY the consensus LDpred2-auto betas (no individual data).
#   2. A filled-in curation template .xlsx skeleton for the Score
#      Metadata sheet (other sheets are left for manual completion).
#   3. Multi-PRS Lasso coefficient TSVs (Feature, Coefficient) extracted
#      from the disease-prediction *_raw_results.pkl files.
#
# All outputs are written to ${OUT_ROOT} and are SAFE to share publicly
# (no UKB individual-level data is included anywhere).
#
# Run as:  sbatch run_pgscat_pipeline.sbatch [/path/to/curation_template.xlsx]
# ===================================================================

set -euo pipefail

exec 2>&1
echo "[INFO] host=$(hostname) date=$(date) job=${SLURM_JOBID:-na}"

# -------- Conda envs --------
CONDA_R_ENV="${CONDA_R_ENV:-/ssu/gassu/conda_envs/bigsnprenv_2025}"
CONDA_PY_ENV="${CONDA_PY_ENV:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

source /home/${USER}/.bashrc

# -------- Paths --------
SCRIPT_DIR="${SCRIPT_DIR:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/PGSCat}"

# Source (paper) inputs:
RDS_DIR="${RDS_DIR:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc}"
RDS_PREFIX="${RDS_PREFIX:-run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_}"
RDS_SUFFIX="${RDS_SUFFIX:-.fullDS.auto.mod.LDPred2.rds}"
BGEN="${BGEN:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/lw_gw_indep/gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas}"
SUMSTATS_COLS="${SUMSTATS_COLS:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/sumcols_UKBB_regenie.json}"
RSIDS_ROOT="${RSIDS_ROOT:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/lw_gw_indep/lw_SNPs_gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4}"

MULTIPRS_RESULTS_DIR="${MULTIPRS_RESULTS_DIR:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/analyses/panCohort_auto_lw_gw_20PC_1000it_caucasiancohort_king0p0625}"

# Curation template (the official PGS Catalog .xlsx). May be passed as $1.
# Defaults to the v11 template bundled in this folder.
CURATION_TEMPLATE="${1:-${CURATION_TEMPLATE:-${SCRIPT_DIR}/PGSSubmission_CurationTemplate_v11.xlsx}}"

# Output:
OUT_ROOT="${OUT_ROOT:-/project/ukbblatent/results/F20208v3_DiffAE/PGSCat_share}"
RAW_DIR="${OUT_ROOT}/scoring_raw"
GZ_DIR="${OUT_ROOT}/scoring_files"
COEF_DIR="${OUT_ROOT}/multiPRS_coefficients"
COMBINED_RAW_DIR="${OUT_ROOT}/combined_scoring_raw"
COMBINED_GZ_DIR="${OUT_ROOT}/combined_scoring_files"
CURATION_OUT="${OUT_ROOT}/filled_curation_template.xlsx"

mkdir -p "${RAW_DIR}" "${GZ_DIR}" "${COEF_DIR}" "${COMBINED_RAW_DIR}" "${COMBINED_GZ_DIR}"

# Publication metadata (placeholders - update before submission)
CITATION="${CITATION:-(manuscript in preparation - update before PGS Catalog submission)}"
PGP_ID="${PGP_ID:-}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# EFO mapping for the 49 per-latent scores. Reviewer-suggested term:
# EFO_0022611 = "magnetic resonance imaging of the heart".
LATENT_TRAIT_EFO="${LATENT_TRAIT_EFO:-EFO:0022611}"
LATENT_TRAIT_MAPPED="${LATENT_TRAIT_MAPPED:-magnetic resonance imaging of the heart}"

# Multi-PRS combination knobs:
FOLD_AGG="${FOLD_AGG:-mean_nz}"
PRS_SCALE_TSV="${PRS_SCALE_TSV:-}"  # optional: latent,mean,sd if model used z-scored PRS

# ===================================================================
# STEP 1 - extract per-latent raw scoring TSVs (R, bigsnpr)
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 1] R extraction: SNP map + final_beta_auto -> raw scoring TSVs"
echo "===================================================================="
conda activate "${CONDA_R_ENV}"

Rscript "${SCRIPT_DIR}/extract_scoring_files.R" \
    --rds_dir       "${RDS_DIR}" \
    --rds_prefix    "${RDS_PREFIX}" \
    --rds_suffix    "${RDS_SUFFIX}" \
    --bgen          "${BGEN}" \
    --sumstats_root "${SUMSTATS_ROOT}" \
    --sumstats_cols "${SUMSTATS_COLS}" \
    --rsids_root    "${RSIDS_ROOT}" \
    --filt_maf      1 \
    --output_dir    "${RAW_DIR}" \
    --threads       "${THREADS}"

conda deactivate

# ===================================================================
# STEP 2 - wrap raw TSVs with PGS Catalog v2.0 header + gzip
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 2] Wrap scoring files with PGS Catalog header + gzip"
echo "===================================================================="
conda activate "${CONDA_PY_ENV}"

python "${SCRIPT_DIR}/wrap_scoring_files.py" \
    --raw_dir      "${RAW_DIR}" \
    --out_dir      "${GZ_DIR}" \
    --citation     "${CITATION}" \
    --pgp_id       "${PGP_ID}" \
    --trait_efo    "${LATENT_TRAIT_EFO}" \
    --trait_mapped "${LATENT_TRAIT_MAPPED}"

# ===================================================================
# STEP 3 - extract multi-PRS Lasso coefficients (safe-to-share subset)
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 3] Multi-PRS Lasso coefficient extraction"
echo "===================================================================="
python "${SCRIPT_DIR}/extract_multiprs_coefficients.py" \
    --results_dir "${MULTIPRS_RESULTS_DIR}" \
    --out_dir     "${COEF_DIR}"

# ===================================================================
# STEP 3b - back-calculate per-disease combined SNP-level scoring files
# (Reviewer request: "single set of SNP-level weights rather than
#  only LASSO coefficients on PRS+covariates".)
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 3b] Combined per-disease SNP-level weights (back-calculated)"
echo "===================================================================="
DERIVE_ARGS=(--results_dir "${MULTIPRS_RESULTS_DIR}" \
             --raw_dir     "${RAW_DIR}" \
             --out_dir     "${COMBINED_RAW_DIR}" \
             --fold_agg    "${FOLD_AGG}")
if [[ -n "${PRS_SCALE_TSV}" && -f "${PRS_SCALE_TSV}" ]]; then
    DERIVE_ARGS+=(--prs_scale_tsv "${PRS_SCALE_TSV}")
fi
python "${SCRIPT_DIR}/derive_combined_pgs.py" "${DERIVE_ARGS[@]}"

# ===================================================================
# STEP 3c - wrap combined per-disease scoring files with PGS header
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 3c] Wrap combined per-disease scoring files"
echo "===================================================================="
if compgen -G "${COMBINED_RAW_DIR}/*.scoring.raw.tsv" > /dev/null; then
    python "${SCRIPT_DIR}/wrap_scoring_files.py" \
        --raw_dir              "${COMBINED_RAW_DIR}" \
        --out_dir              "${COMBINED_GZ_DIR}" \
        --citation             "${CITATION}" \
        --pgp_id               "${PGP_ID}" \
        --name_strip_suffix    "_combined" \
        --trait_reported_prefix "Multi-PRS combined risk score for " \
        --method_name          "LDpred2-auto + LASSO multi-PRS combination" \
        --method_params        "per-latent LDpred2-auto betas (49 latents) linearly combined using disease-specific LASSO coefficients aggregated across CV folds (${FOLD_AGG}); covariate terms NOT folded in - see *_covariate_coefficients.tsv sidecar"
else
    echo "[WARN] no combined scoring files produced in ${COMBINED_RAW_DIR}"
fi

# ===================================================================
# STEP 4 - fill curation template .xlsx (Score Metadata sheet)
# ===================================================================
echo
echo "===================================================================="
echo "[STEP 4] Build curation template skeleton"
echo "===================================================================="
if [[ -f "${CURATION_TEMPLATE}" ]]; then
    python "${SCRIPT_DIR}/build_curation_xlsx.py" \
        --template     "${CURATION_TEMPLATE}" \
        --raw_dir      "${RAW_DIR}" \
        --combined_dir "${COMBINED_RAW_DIR}" \
        --out          "${CURATION_OUT}" \
        --pubmed_id    "${PUBMED_ID:-}" \
        --doi          "${PUB_DOI:-}" \
        --journal      "${PUB_JOURNAL:-}" \
        --pub_date     "${PUB_DATE:-}" \
        --author_last  "${PUB_AUTHOR_LAST:-Chatterjee}" \
        --author_init  "${PUB_AUTHOR_INIT:-S}" \
        --gwas_pmid_doi "${CITATION}" \
        --gwas_n       "${GWAS_N:-0}" \
        --train_n      "${TRAIN_N:-0}" \
        --test_n       "${TEST_N:-0}"
else
    echo "[WARN] curation template not found at: ${CURATION_TEMPLATE}"
    echo "       Download it from https://www.pgscatalog.org/submit/ and re-run only step 4:"
    echo "       python ${SCRIPT_DIR}/build_curation_xlsx.py \\"
    echo "           --template /path/to/curation_template.xlsx \\"
    echo "           --raw_dir  ${RAW_DIR} \\"
    echo "           --out      ${CURATION_OUT}"
fi

conda deactivate

echo
echo "===================================================================="
echo "[DONE] All outputs in: ${OUT_ROOT}"
echo "  - Per-latent scoring files:       ${GZ_DIR}/*.txt.gz"
echo "  - Combined per-disease scoring:   ${COMBINED_GZ_DIR}/*.txt.gz"
echo "  - Combined covariate sidecars:    ${COMBINED_RAW_DIR}/*_covariate_coefficients.tsv"
echo "  - Multi-PRS Lasso coefficients:   ${COEF_DIR}/"
echo "  - Curation template (review):     ${CURATION_OUT}"
echo "===================================================================="

