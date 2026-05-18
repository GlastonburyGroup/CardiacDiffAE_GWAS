#!/bin/bash
#SBATCH --job-name=gwascat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq
#SBATCH --time=12-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=gwascat_%x_%j.log

# ===================================================================
# GWAS Catalog deposition pipeline for DiffAE4CardiacGWAS.
# Driven entirely from manifests/inputs.yaml: all input paths, sample
# sizes, ancestries, covariates and output locations are read from
# that file.
#
# Run as:  sbatch master_pipeline.sh
# ===================================================================

set -euo pipefail

exec 2>&1
echo "[INFO] host=$(hostname) date=$(date) job=${SLURM_JOBID:-na}"

# -------- Conda env --------
CONDA_PY_ENV="${CONDA_PY_ENV:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

# Script directory — set explicitly so sbatch --chdir does not break HERE.
SCRIPT_DIR="${SCRIPT_DIR:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/GWASCat}"

source /home/${USER}/.bashrc
conda activate "${CONDA_PY_ENV}"

# Resolve script directory robustly even when run via sbatch --chdir.
# BASH_SOURCE[0] may be a bare filename when the working directory has been
# changed by the scheduler before the shell is launched.
HERE="${SCRIPT_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)}"
cd "${HERE}"

INPUTS="${HERE}/manifests/inputs.yaml"
if [ ! -f "${INPUTS}" ]; then
    echo "ERROR: ${INPUTS} not found." >&2
    exit 1
fi

# We use `yq` (Python wrapper or the Go binary works) to parse YAML.
if ! command -v yq >/dev/null 2>&1; then
    echo "ERROR: 'yq' is required (pip install yq, or install the Go binary)." >&2
    exit 1
fi

# Helpers -------------------------------------------------------------------

yqr() { yq -r "$1" "${INPUTS}"; }

OUTROOT="$(yqr '.output_root')"
MAPPING="$(yqr '.name_mapping')"
GENE_COORDS="$(yqr '.gene_coords' | sed "s|\${output_root}|${OUTROOT}|g")"
BASE_COV="$(yqr '.covariates_base' | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')"

echo "============================================"
echo "  GWAS Catalog Deposition Pipeline"
echo "  DiffAE4CardiacGWAS"
echo "  $(date)"
echo "  Output root: ${OUTROOT}"
echo "============================================"
echo ""

mkdir -p \
    "${OUTROOT}/variant_level/GWAS_Discovery" \
    "${OUTROOT}/variant_level/GWAS_Replication" \
    "${OUTROOT}/variant_level/GWAS_Sex_Interaction" \
    "${OUTROOT}/variant_level/EWAS_Discovery" \
    "${OUTROOT}/gene_level/Gene_Burden_Discovery" \
    "${OUTROOT}/gene_level/SKATO_ACAT_Discovery" \
    "${OUTROOT}/metadata" \
    "${OUTROOT}/logs" \
    "${OUTROOT}/manifests" \
    "${OUTROOT}/globus_stage"

# Stage the mapping TSV under the output root so each run has a self-contained
# manifests/ directory.
if [ -f "${MAPPING}" ]; then
    cp -f "${MAPPING}" "${OUTROOT}/manifests/Z_mapping_FINAL.tsv"
else
    echo "ERROR: Mapping file ${MAPPING} not found." >&2
    exit 1
fi
MAPPING_LOCAL="${OUTROOT}/manifests/Z_mapping_FINAL.tsv"

# Optional gene coordinates (warn if not yet generated; gene step uses GENPOS).
if [ -f "${GENE_COORDS}" ]; then
    echo "Gene coords: ${GENE_COORDS}"
    GENE_COORDS_FLAG="--gene-coords ${GENE_COORDS}"
else
    echo "WARNING: gene_coords.tsv not found at ${GENE_COORDS}."
    echo "Generate from Ensembl 85 GTF (see GWAS_CATALOG_DEPOSITION_GUIDE.md)."
    GENE_COORDS_FLAG=""
fi
echo ""

# Step 0: verify each input directory exists with the expected file count.
echo "--- Step 0: Verifying inputs ---"
verify_dir () {
    local key="$1"
    local section="$2"
    local indir
    indir="$(yqr ".${section}.${key}.input_dir")"
    local expected
    expected="$(yqr ".${section}.${key}.expected_files")"
    if [ ! -d "${indir}" ]; then
        echo "ERROR: ${section}.${key} input dir not found: ${indir}" >&2
        exit 1
    fi
    local actual
    actual=$(find "${indir}" -maxdepth 1 -name '*.gz' | wc -l)
    echo "  ${section}.${key}: ${actual} files (expected ${expected})"
    if [ "${actual}" -lt "${expected}" ]; then
        echo "ERROR: file count below expected." >&2
        exit 1
    fi
}
for k in GWAS_Discovery GWAS_Replication GWAS_Sex_Interaction EWAS_Discovery; do
    verify_dir "${k}" "variant_analyses"
done
for k in Gene_Burden_Discovery SKATO_ACAT_Discovery; do
    verify_dir "${k}" "gene_analyses"
done
echo ""

# Convenience: run a single variant-level conversion.
run_variant () {
    local key="$1"
    local section="variant_analyses"
    local indir n ancestry adesc atype extra maf gbuild asw
    indir="$(yqr ".${section}.${key}.input_dir")"
    n="$(yqr ".${section}.${key}.sample_size")"
    ancestry="$(yqr ".${section}.${key}.ancestry")"
    adesc="$(yqr ".${section}.${key}.ancestry_description")"
    atype="$(yqr ".${section}.${key}.analysis_type")"
    extra="$(yqr ".${section}.${key}.extra_covariates" | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')"
    maf="$(yqr ".${section}.${key}.maf_filter // \"\"")"
    gbuild="$(yqr ".${section}.${key}.genome_build")"
    asw="$(yqr ".${section}.${key}.analysis_software")"

    local maf_flag=""
    if [ -n "${maf}" ] && [ "${maf}" != "null" ]; then
        maf_flag="--maf-filter ${maf}"
    fi

    echo "  >>> Variant: ${key} (N=${n}, build=${gbuild}, sw=${asw})"
    python convert_variant_level.py \
        --input-dir "${indir}" \
        --output-dir "${OUTROOT}/variant_level/${key}" \
        --analysis-type "${atype}" \
        --sample-size "${n}" \
        --ancestry "${ancestry}" \
        --ancestry-description "${adesc}" \
        --covariates "${BASE_COV}" \
        --extra-covariates "${extra}" \
        --name-mapping "${MAPPING_LOCAL}" \
        --genome-build "${gbuild}" \
        --analysis-software "${asw}" \
        ${maf_flag} \
        2>&1 | tee "${OUTROOT}/logs/convert_${key}.log"
}

# Convenience: run a single gene-level conversion.
run_gene () {
    local key="$1"
    local section="gene_analyses"
    local indir n ancestry adesc atype extra gbuild asw
    indir="$(yqr ".${section}.${key}.input_dir")"
    n="$(yqr ".${section}.${key}.sample_size")"
    ancestry="$(yqr ".${section}.${key}.ancestry")"
    adesc="$(yqr ".${section}.${key}.ancestry_description")"
    atype="$(yqr ".${section}.${key}.analysis_type")"
    extra="$(yqr ".${section}.${key}.extra_covariates" | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')"
    gbuild="$(yqr ".${section}.${key}.genome_build")"
    asw="$(yqr ".${section}.${key}.analysis_software")"

    echo "  >>> Gene: ${key} (N=${n}, build=${gbuild}, sw=${asw})"
    python convert_gene_level.py \
        --input-dir "${indir}" \
        --output-dir "${OUTROOT}/gene_level/${key}" \
        --analysis-type "${atype}" \
        --sample-size "${n}" \
        --ancestry "${ancestry}" \
        --ancestry-description "${adesc}" \
        --covariates "${BASE_COV}" \
        --extra-covariates "${extra}" \
        --name-mapping "${MAPPING_LOCAL}" \
        --genome-build "${gbuild}" \
        --analysis-software "${asw}" \
        ${GENE_COORDS_FLAG} \
        2>&1 | tee "${OUTROOT}/logs/convert_${key}.log"
}

# --------------------------------------------------------------------------
# Step 1: variant-level conversions (4 analyses)
# --------------------------------------------------------------------------
echo "--- Step 1: Converting variant-level files ---"
for k in GWAS_Discovery GWAS_Replication GWAS_Sex_Interaction EWAS_Discovery; do
    run_variant "${k}"
    echo ""
done

# --------------------------------------------------------------------------
# Step 2: gene-level conversions (2 analyses)
# --------------------------------------------------------------------------
echo "--- Step 2: Converting gene-level files ---"
for k in Gene_Burden_Discovery SKATO_ACAT_Discovery; do
    run_gene "${k}"
    echo ""
done

# --------------------------------------------------------------------------
# Step 3: aggregate conversion logs
# --------------------------------------------------------------------------
echo "--- Step 3: Aggregating conversion logs ---"
SUMMARY="${OUTROOT}/logs/conversion_summary.csv"
echo "analysis,total,ok,errors" > "${SUMMARY}"
for d in "${OUTROOT}/variant_level"/* "${OUTROOT}/gene_level"/*; do
    log="${d}/conversion_log.csv"
    [ -f "${log}" ] || continue
    name="$(basename "${d}")"
    total=$(($(wc -l < "${log}") - 1))
    ok=$(grep -c ',OK,' "${log}" || true)
    errs=$((total - ok))
    echo "${name},${total},${ok},${errs}" >> "${SUMMARY}"
done
column -ts, "${SUMMARY}"
if awk -F, 'NR>1 && $4>0 {found=1} END {exit !found}' "${SUMMARY}"; then
    echo "ERROR: some conversions failed; see ${SUMMARY}" >&2
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
# Step 4: validate variant-level files
# --------------------------------------------------------------------------
echo "--- Step 4: Validating variant-level files ---"
OUTROOT="${OUTROOT}" bash validate_variant_level.sh \
    2>&1 | tee "${OUTROOT}/logs/validation.log"
echo ""

# --------------------------------------------------------------------------
# Step 5: spot-check
# --------------------------------------------------------------------------
echo "--- Step 5: Spot-checking converted files ---"
python spotcheck.py \
    --output-root "${OUTROOT}" \
    --seed 0 \
    2>&1 | tee "${OUTROOT}/logs/spotcheck.txt"
echo ""

# --------------------------------------------------------------------------
# Step 6: metadata spreadsheet
# --------------------------------------------------------------------------
echo "--- Step 6: Generating metadata spreadsheet ---"
python generate_metadata.py \
    --inputs "${INPUTS}" \
    --output "${OUTROOT}/metadata/metadata_submission.xlsx"
echo ""

# --------------------------------------------------------------------------
# Step 7: copy the cover-letter template into the output tree
# --------------------------------------------------------------------------
echo "--- Step 7: Staging cover letter ---"
cp -f "${HERE}/cover_letter.md" "${OUTROOT}/metadata/cover_letter.md"
echo ""

conda deactivate

echo "============================================"
echo "  Pipeline complete - $(date)"
echo "============================================"
echo "Output:    ${OUTROOT}"
echo "Metadata:  ${OUTROOT}/metadata/metadata_submission.xlsx"
echo "Cover:     ${OUTROOT}/metadata/cover_letter.md"
echo ""
echo "Next steps:"
echo "  1. Review ${OUTROOT}/logs/conversion_summary.csv (must be 0 errors)"
echo "  2. Review ${OUTROOT}/logs/validation.log"
echo "  3. Stage to Globus: ${OUTROOT}/globus_stage/"
echo "  4. Submit at https://www.ebi.ac.uk/gwas/deposition"
echo "  5. Email gwas-subs@ebi.ac.uk for manual gene-level processing"
echo ""
