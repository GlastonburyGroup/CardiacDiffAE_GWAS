#!/usr/bin/env python3
"""
convert_variant_level.py

Batch-convert REGENIE variant-level output to GWAS-SSF v1.0 format.

Handles all four variant-level analysis types:

  GWAS Discovery / Replication (14 cols):
    CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA

  GWAS Sex Interaction (15 cols):
    CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA MAF
    TEST = "ADD-INT_SNPxSex=1.0", already filtered upstream to MAF < 0.01.

  EWAS Discovery (13 cols, no INFO):
    CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE CHISQ LOG10P EXTRA

Phenotype names are remapped from the REGENIE file stem (e.g. "S1701_Z1")
to the publication-facing identifier (e.g. "Z1_S1") via --name-mapping.
The output filenames, embedded sumstats and YAML metadata all use the
mapped identifier.

Usage:
    python convert_variant_level.py \\
        --input-dir <REGENIE final_sumstats dir> \\
        --output-dir <out dir> \\
        --analysis-type "GWAS Discovery" \\
        --sample-size 47740 \\
        --name-mapping /path/to/Z_mapping_FINAL.tsv
"""

import argparse
import glob
import hashlib
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# Standard seven-covariate string (verbatim what we used in REGENIE).
BASE_COVARIATES = (
    "genetic sex, age at acquisition, genotyping batch, "
    "body surface area (BSA), imaging year, assessment centre, visit number"
)

# Suffix appended to every reported_trait / trait_description.
DIFFAE_NOTE = "(data-driven DiffAE latent; no a-priori biological annotation)"


def load_name_mapping(path):
    """Return dict[file_stem] -> publication_id from Z_mapping_FINAL.tsv.

    Expected TSV columns (with header): Latent, x
      - Latent = publication-facing ID (e.g. "Z1_S1")
      - x      = REGENIE file stem    (e.g. "S1701_Z1")
    """
    if path is None:
        return None
    df = pd.read_csv(path, sep="\t")
    if not {"Latent", "x"}.issubset(df.columns):
        raise ValueError(
            f"Mapping file {path} must contain 'Latent' and 'x' columns; "
            f"found {list(df.columns)}"
        )
    mapping = dict(zip(df["x"].astype(str), df["Latent"].astype(str)))
    # Allow round-trip: if a phenotype was already supplied in publication form,
    # map it to itself so the converter never silently rewrites it.
    for pub_id in df["Latent"].astype(str):
        mapping.setdefault(pub_id, pub_id)
    return mapping


def map_phenotype(file_stem, mapping, strict=True):
    """Look up the publication-facing latent ID for a REGENIE file stem."""
    if mapping is None:
        return file_stem
    if file_stem in mapping:
        return mapping[file_stem]
    if strict:
        raise KeyError(
            f"Phenotype '{file_stem}' is not present in the name-mapping TSV. "
            f"Refusing to write an unmapped latent."
        )
    return file_stem


def log10p_to_pvalue(log10p_series):
    """Convert -log10(p) to p-value, preserving precision for extreme tails."""
    pvals = []
    for val in log10p_series:
        if pd.isna(val):
            pvals.append(np.nan)
        elif val > 300:
            # Avoid underflow: write as mantissa e-exponent in scientific form.
            mantissa = 10 ** (-(val - int(val)))
            pvals.append(f"{mantissa:.6e}".replace("e+00", f"e-{int(val)}"))
        else:
            pvals.append(10 ** (-val))
    return pvals


def compute_md5(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def assert_maf_filtered(df, threshold, file_stem, conversion_log):
    """For sex-interaction inputs, confirm the MAF column is already < threshold.

    Writes a soft warning into the conversion log if the column is missing,
    so we never silently skip the check. Raises if the assertion is violated.
    """
    if "MAF" not in df.columns:
        conversion_log.append(
            f"WARN: {file_stem}: MAF column absent; cannot verify MAF < {threshold}"
        )
        return
    max_maf = df["MAF"].dropna().max()
    if pd.notna(max_maf) and max_maf >= threshold:
        raise AssertionError(
            f"{file_stem}: MAF.max() = {max_maf} >= {threshold}; "
            f"sex-interaction sumstats expected pre-filtered."
        )


def convert_single_file(filepath, output_dir, pub_id, meta, soft_log):
    """Convert one REGENIE file to a GWAS-SSF .tsv.gz + .tsv.gz-meta.yaml pair."""
    df = pd.read_csv(filepath, sep=r"\s+", comment="#")

    # Each REGENIE step-2 output already contains exactly one test type per
    # file. We never filter on TEST == "ADD" here because the sex-interaction
    # files use TEST = "ADD-INT_SNPxSex=1.0" and would be dropped entirely.
    if "TEST" in df.columns:
        tests = df["TEST"].unique()
        if len(tests) > 1:
            df = df[df["TEST"] == tests[0]].copy()

    # Optional MAF sanity check (sex-interaction).
    if meta.get("maf_filter") is not None:
        assert_maf_filtered(df, meta["maf_filter"], pub_id, soft_log)

    # Convert LOG10P to p-value
    if "LOG10P" in df.columns:
        df["P"] = log10p_to_pvalue(df["LOG10P"])

    # Build GWAS-SSF (mandatory columns in required order)
    ssf = pd.DataFrame()
    ssf["chromosome"] = df["CHROM"].astype(str)
    ssf["base_pair_location"] = df["GENPOS"].astype(int)
    ssf["effect_allele"] = df["ALLELE1"]
    ssf["other_allele"] = df["ALLELE0"]
    ssf["beta"] = df["BETA"]
    ssf["standard_error"] = df["SE"]
    ssf["effect_allele_frequency"] = df["A1FREQ"]
    ssf["p_value"] = df["P"]

    # Recommended columns
    ssf["variant_id"] = df["ID"]
    if "N" in df.columns:
        ssf["n"] = df["N"].astype(int)
    if "INFO" in df.columns:
        ssf["info"] = df["INFO"]

    # Sort by chromosome then position.
    chrom_order = {str(i): i for i in range(1, 23)}
    chrom_order.update({"X": 23, "Y": 24, "MT": 25})
    ssf["_s"] = ssf["chromosome"].map(chrom_order).fillna(99)
    ssf = ssf.sort_values(["_s", "base_pair_location"]).drop(columns=["_s"])
    ssf = ssf.dropna(subset=["chromosome", "base_pair_location", "p_value"])

    # Write the sumstats (filename uses the publication-facing latent ID).
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", pub_id)
    out_tsv = Path(output_dir) / f"{safe}.tsv.gz"
    ssf.to_csv(out_tsv, sep="\t", index=False, na_rep="NA", compression="gzip")

    md5 = compute_md5(out_tsv)
    out_yaml = Path(output_dir) / f"{safe}.tsv.gz-meta.yaml"
    trait = (
        f"Cardiac MRI DiffAE latent {pub_id} - "
        f"{meta['analysis_type']} {DIFFAE_NOTE}"
    )
    with open(out_yaml, "w") as f:
        f.write(
            f"date_metadata_last_modified: {date.today().isoformat()}\n"
            f"genome_assembly: {meta['genome_build']}\n"
            f"coordinate_system: 1-based\n"
            f"data_file_name: {out_tsv.name}\n"
            f"file_type: GWAS-SSF v1.0\n"
            f"data_file_md5sum: {md5}\n"
            f"sample_size: {meta['sample_size']}\n"
            f"sample_ancestry_category: {meta['ancestry']}\n"
            f"sample_ancestry_description: \"{meta['ancestry_description']}\"\n"
            f"trait_description: \"{trait}\"\n"
            f"analysis_software: {meta['analysis_software']}\n"
            f"adjusted_covariates: \"{meta['covariates']}\"\n"
            f"is_harmonised: false\n"
            f"is_sorted: true\n"
        )
    return len(ssf)


def batch_convert(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(str(Path(args.input_dir) / "*.regenie.gz")))
    if not files:
        files = sorted(glob.glob(str(Path(args.input_dir) / "*.gz")))
    if not files:
        print(f"ERROR: No .regenie.gz files in {args.input_dir}")
        sys.exit(1)

    mapping = load_name_mapping(args.name_mapping)

    # Effective covariate string = base seven + analysis-specific extras.
    cov = args.covariates
    if args.extra_covariates:
        cov = f"{cov}, {args.extra_covariates}"

    meta = {
        "genome_build": args.genome_build,
        "sample_size": args.sample_size,
        "ancestry": args.ancestry,
        "ancestry_description": args.ancestry_description,
        "analysis_type": args.analysis_type,
        "analysis_software": args.analysis_software,
        "covariates": cov,
        "maf_filter": args.maf_filter,
    }

    print(f"Analysis: {args.analysis_type}")
    print(f"Input:    {args.input_dir} ({len(files)} files)")
    print(f"Output:   {args.output_dir}")
    print(f"Mapping:  {args.name_mapping or '<none>'}")
    print(
        f"Build:    {args.genome_build}  N: {args.sample_size}  "
        f"Ancestry: {args.ancestry}"
    )
    print(f"Covariates: {cov}")
    print("---")

    results = []
    soft_log = []
    for i, fp in enumerate(files, 1):
        bn = Path(fp).name
        file_stem = bn.split(".")[0]
        try:
            pub_id = map_phenotype(file_stem, mapping, strict=mapping is not None)
        except KeyError as e:
            print(f"  [{i:3d}/{len(files)}] {file_stem}... MAPPING ERROR: {e}")
            results.append({
                "file": bn, "file_stem": file_stem, "pub_id": "",
                "status": f"MAPPING ERROR: {e}", "n": 0,
            })
            continue

        print(
            f"  [{i:3d}/{len(files)}] {file_stem} -> {pub_id}... ",
            end="", flush=True,
        )
        try:
            n = convert_single_file(fp, args.output_dir, pub_id, meta, soft_log)
            print(f"OK ({n:,} variants)")
            results.append({
                "file": bn, "file_stem": file_stem, "pub_id": pub_id,
                "status": "OK", "n": n,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": bn, "file_stem": file_stem, "pub_id": pub_id,
                "status": f"ERROR: {e}", "n": 0,
            })

    pd.DataFrame(results).to_csv(
        Path(args.output_dir) / "conversion_log.csv", index=False
    )
    if soft_log:
        with open(Path(args.output_dir) / "conversion_warnings.txt", "w") as f:
            f.write("\n".join(soft_log) + "\n")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n--- {ok}/{len(files)} converted ---")
    if ok != len(files):
        sys.exit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--analysis-type", required=True)
    p.add_argument("--genome-build", default="GRCh38")
    p.add_argument("--sample-size", type=int, required=True)
    p.add_argument("--ancestry", default="European")
    p.add_argument(
        "--ancestry-description",
        default="UK Biobank Caucasian individuals passing phenotypic and genotype QC",
    )
    p.add_argument("--analysis-software", default="REGENIE v3.4.1")
    p.add_argument(
        "--covariates", default=BASE_COVARIATES,
        help="Base covariate list. Defaults to the seven covariates used for "
             "GWAS Discovery / Replication.",
    )
    p.add_argument(
        "--extra-covariates", default="",
        help="Analysis-specific covariates appended to --covariates "
             "(e.g. 'SNP x genetic-sex interaction term' or "
             "'WES release tranche (UKB field 32050)').",
    )
    p.add_argument(
        "--name-mapping", default=None,
        help="TSV with columns 'Latent' (publication ID, Zx_Sy) and 'x' "
             "(REGENIE file stem). Required for production runs.",
    )
    p.add_argument(
        "--maf-filter", type=float, default=None,
        help="If set, assert that the MAF column max < this threshold. "
             "Use 0.01 for sex-interaction sumstats (already filtered upstream).",
    )
    batch_convert(p.parse_args())
