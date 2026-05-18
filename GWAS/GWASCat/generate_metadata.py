#!/usr/bin/env python3
"""
generate_metadata.py

Generate the GWAS Catalog submission metadata spreadsheet for all six
analyses (4 variant-level + 2 gene-level), driven by inputs.yaml.

Scans each converted output directory for .tsv.gz files and creates
one row per study in the metadata spreadsheet. Every reported_trait /
study_description carries the publication-facing latent ID (Zx_Sy) and
the DiffAE annotation note.

Usage:
    python generate_metadata.py \\
        --inputs manifests/inputs.yaml \\
        --output metadata/metadata_submission.xlsx
"""

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
import yaml


DIFFAE_NOTE = "(data-driven DiffAE latent; no a-priori biological annotation)"


def expand_vars(value, ctx):
    """Resolve ${var} references using earlier keys in the same YAML file."""
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda m: str(ctx.get(m.group(1), m.group(0))), value)
    return value


def load_inputs(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Two-pass expansion so e.g. gene_coords: ${output_root}/manifests/... resolves.
    flat = {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
    for k, v in raw.items():
        raw[k] = expand_vars(v, flat) if isinstance(v, str) else v
    return raw


def get_files(directory):
    """Return sorted list of .tsv.gz filenames in a directory (no logs)."""
    files = sorted(glob.glob(str(Path(directory) / "*.tsv.gz")))
    return [Path(f).name for f in files if "conversion_log" not in f]


def covariate_string(base, extra):
    if extra:
        return f"{base}, {extra}"
    return base


def latent_from_filename(fname, suffix):
    """Strip the analysis-specific suffix from an output filename to get the
    latent ID. For variant files this is just the stem; for gene files we
    strip _burden_*_M*_0_01.tsv.gz."""
    stem = fname.replace(".tsv.gz", "")
    if suffix == "variant":
        return stem
    # Gene-level: stem is "Z1_S1_burden_<test>_M{1,2,3}_<aaf>"
    m = re.match(r"^(?P<lat>.+?)_burden_(?P<test>[a-z_]+)_M(?P<mask>\d+)_(?P<aaf>.+)$", stem)
    if m:
        return m.group("lat"), m.group("test"), f"M{m.group('mask')}", m.group("aaf")
    return stem, "", "", ""


def generate(inputs_path, output_path):
    cfg = load_inputs(inputs_path)
    base_cov = " ".join(cfg["covariates_base"].split())  # collapse YAML folded whitespace
    output_root = cfg["output_root"]

    all_rows = []

    # Variant-level analyses
    for name, a in cfg["variant_analyses"].items():
        out_dir = Path(output_root) / "variant_level" / name
        filenames = get_files(out_dir)
        if not filenames:
            print(f"  WARNING: No files in {out_dir}")
            continue
        cov = covariate_string(base_cov, a.get("extra_covariates", "").strip())
        gbuild = a.get("genome_build", cfg.get("genome_build", "GRCh38"))
        asw = a.get("analysis_software", cfg.get("analysis_software", "REGENIE v3.4.1"))
        for fname in filenames:
            latent = latent_from_filename(fname, "variant")
            all_rows.append({
                "study_tag": f"{a['study_prefix']}_{latent}",
                "reported_trait": (
                    f"Cardiac MRI DiffAE latent {latent} - "
                    f"{a['analysis_type']} {DIFFAE_NOTE}"
                ),
                "efo_trait": "",   # blank; curated post-PMID
                "background_trait": "",
                "study_type": "Quantitative trait",
                "study_description": (
                    f"Cardiac MRI DiffAE latent {latent}; {a['analysis_type']}. "
                    f"UK Biobank, REGENIE v3.4.1, {gbuild}. {DIFFAE_NOTE}"
                ),
                "sample_description": (
                    f"{a['sample_size']:,} individuals from UK Biobank "
                    f"({a['ancestry_description']})"
                ),
                "sample_size": a["sample_size"],
                "sample_ancestry": a["ancestry"],
                "sample_ancestry_description": a["ancestry_description"],
                "cohort": "UK Biobank",
                "summary_statistics_file": fname,
                "genome_assembly": gbuild,
                "genotyping_technology": a["genotyping_technology"],
                "analysis_software": asw,
                "adjusted_covariates": cov,
                "proposed_efo": cfg.get("efo_proposed", ""),
            })
        print(f"  {a['analysis_type']:<35s} {len(filenames):>4d} studies")

    # Gene-level analyses
    for name, a in cfg["gene_analyses"].items():
        out_dir = Path(output_root) / "gene_level" / name
        filenames = get_files(out_dir)
        if not filenames:
            print(f"  WARNING: No files in {out_dir}")
            continue
        cov = covariate_string(base_cov, a.get("extra_covariates", "").strip())
        gbuild = a.get("genome_build", cfg.get("genome_build", "GRCh38"))
        asw = a.get("analysis_software", cfg.get("analysis_software", "REGENIE v3.4.1"))
        for fname in filenames:
            latent, test_slug, mask, aaf = latent_from_filename(fname, "gene")
            aaf_pretty = aaf.replace("_", ".") if aaf else ""
            all_rows.append({
                "study_tag": f"{a['study_prefix']}_{latent}_{mask}_{aaf}".strip("_"),
                "reported_trait": (
                    f"Cardiac MRI DiffAE latent {latent} - {a['analysis_type']} "
                    f"(mask {mask}, AAF<{aaf_pretty}) {DIFFAE_NOTE}"
                ),
                "efo_trait": "",
                "background_trait": "",
                "study_type": "Quantitative trait (gene-level rare variant test)",
                "study_description": (
                    f"Cardiac MRI DiffAE latent {latent}; {a['analysis_type']}, "
                    f"mask {mask}, AAF<{aaf_pretty}. "
                    f"UK Biobank WES, REGENIE v3.4.1 gene-based, {gbuild}. {DIFFAE_NOTE}"
                ),
                "sample_description": (
                    f"{a['sample_size']:,} individuals from UK Biobank "
                    f"({a['ancestry_description']})"
                ),
                "sample_size": a["sample_size"],
                "sample_ancestry": a["ancestry"],
                "sample_ancestry_description": a["ancestry_description"],
                "cohort": "UK Biobank",
                "summary_statistics_file": fname,
                "genome_assembly": gbuild,
                "genotyping_technology": a["genotyping_technology"],
                "analysis_software": asw,
                "adjusted_covariates": cov,
                "proposed_efo": cfg.get("efo_proposed", ""),
            })
        print(f"  {a['analysis_type']:<35s} {len(filenames):>4d} studies")

    df = pd.DataFrame(all_rows)
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name="studies")
    print(f"\nMetadata written to {output_path}")
    print(f"Total studies: {len(all_rows)}")

    # Sanity assertion mirrored in PHASE-E verification.
    if len(all_rows):
        assert df["adjusted_covariates"].str.contains("body surface area").all(), \
            "Covariate string lost 'body surface area' in some rows."


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="manifests/inputs.yaml")
    p.add_argument("--output", default="metadata/metadata_submission.xlsx")
    args = p.parse_args()
    generate(args.inputs, args.output)
