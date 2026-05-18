#!/usr/bin/env bash
# validate_variant_level.sh
#
# Validate every converted variant-level GWAS-SSF file under OUTROOT.
# Requires: pip install gwas-sumstats-tools
#
# Usage:
#   OUTROOT=/path/to/output_root bash validate_variant_level.sh

set -euo pipefail

OUTROOT="${OUTROOT:-/project/ukbblatent/results/F20208v3_DiffAE/GWASCat_share}"
OUTBASE="${OUTROOT}/variant_level"
LOG_DIR="${OUTROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG="${LOG_DIR}/validation.log"
FAILURES="${LOG_DIR}/validation_failures.tsv"

echo "GWAS-SSF Validation - $(date)" > "${LOG}"
printf 'analysis\tfile\n' > "${FAILURES}"

TOTAL_PASS=0
TOTAL_FAIL=0

for subdir in GWAS_Discovery GWAS_Replication GWAS_Sex_Interaction EWAS_Discovery; do
    dir="${OUTBASE}/${subdir}"
    [ -d "${dir}" ] || continue
    echo "=== ${subdir} ===" | tee -a "${LOG}"
    pass=0
    fail=0
    for f in "${dir}"/*.tsv.gz; do
        [ -f "${f}" ] || continue
        name=$(basename "${f}")
        if gwas-ssf validate "${f}" 2>&1 | grep -qi "valid\|pass\|success"; then
            pass=$((pass + 1))
        else
            echo "  FAIL: ${name}" | tee -a "${LOG}"
            printf '%s\t%s\n' "${subdir}" "${name}" >> "${FAILURES}"
            fail=$((fail + 1))
        fi
    done
    echo "  ${pass} passed, ${fail} failed" | tee -a "${LOG}"
    TOTAL_PASS=$((TOTAL_PASS + pass))
    TOTAL_FAIL=$((TOTAL_FAIL + fail))
done

echo "" | tee -a "${LOG}"
echo "TOTAL: ${TOTAL_PASS} passed, ${TOTAL_FAIL} failed" | tee -a "${LOG}"

if [ "${TOTAL_FAIL}" -gt 0 ]; then
    echo "WARNING: ${TOTAL_FAIL} files failed validation. See ${FAILURES}." | tee -a "${LOG}"
    exit 1
fi
