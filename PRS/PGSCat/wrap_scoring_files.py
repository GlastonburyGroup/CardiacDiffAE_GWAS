#!/usr/bin/env python
"""wrap_scoring_files.py

Take per-latent raw scoring TSVs produced by ``extract_scoring_files.R``
and wrap them with a PGS Catalog v2.0 formatted header, then gzip them.

Output naming: ``<latent>.txt.gz`` inside ``--out_dir``.

PGS Catalog scoring file format spec (v2.0):
https://www.pgscatalog.org/downloads/#scoring_columns
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd


# PGS Catalog v2.0 scoring file header layout, matching the format
# of files served at https://www.pgscatalog.org/downloads/.
PGS_SECTION = [
    "pgs_id", "pgs_name", "trait_reported", "trait_mapped",
    "trait_efo", "genome_build", "variants_number", "weight_type",
]
SOURCE_SECTION = ["pgp_id", "citation", "license"]
HARMONISATION_SECTION = ["HmPOS_build", "HmPOS_date"]


def build_header(meta: dict) -> str:
    """Return a PGS Catalog v2.0 scoring file header (``#``-prefixed).

    The order is: banner -> ``format_version`` -> POLYGENIC SCORE section
    -> SOURCE section -> (optional) HARMONIZATION section.  Method
    metadata (``pgs_method_name``/``pgs_method_params``) is appended at
    the end of the SOURCE section, which is the slot curators preserve.
    """
    lines: list[str] = [
        "###PGS CATALOG SCORING FILE - see https://www.pgscatalog.org/downloads/#dl_ftp_scoring",
        f"#format_version={meta.get('format_version', '2.0')}",
        "##POLYGENIC SCORE (PGS) INFORMATION",
    ]
    for key in PGS_SECTION:
        val = meta.get(key, "")
        if val is None:
            val = ""
        lines.append(f"#{key}={val}")
    lines.append("##SOURCE INFORMATION")
    for key in SOURCE_SECTION:
        val = meta.get(key, "")
        if val is None or val == "":
            continue
        lines.append(f"#{key}={val}")
    if meta.get("pgs_method_name"):
        lines.append(f"#pgs_method_name={meta['pgs_method_name']}")
    if meta.get("pgs_method_params"):
        lines.append(f"#pgs_method_params={meta['pgs_method_params']}")
    if meta.get("HmPOS_build"):
        lines.append("##HARMONIZATION DETAILS")
        for key in HARMONISATION_SECTION:
            lines.append(f"#{key}={meta.get(key, '')}")
    return "\n".join(lines) + "\n"


def wrap_one(raw_tsv: Path, out_path: Path, meta: dict) -> int:
    df = pd.read_csv(raw_tsv, sep="\t")
    required = {"chr_name", "chr_position", "effect_allele", "effect_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{raw_tsv}: missing required columns {missing}")
    meta = dict(meta)
    meta["variants_number"] = len(df)
    header = build_header(meta)
    with gzip.open(out_path, "wt", compresslevel=6) as fh:
        fh.write(header)
        df.to_csv(fh, sep="\t", index=False)
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_dir", required=True,
                   help="Directory of *.scoring.raw.tsv files")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write gzipped scoring files into")
    p.add_argument("--genome_build", default="GRCh37")
    p.add_argument("--weight_type", default="beta")
    p.add_argument("--license",
                   default="Creative Commons Attribution 4.0 International (CC BY 4.0)")
    p.add_argument("--citation", default="",
                   help="Free-text citation/DOI placeholder for the publication")
    p.add_argument("--pgp_id", default="",
                   help="PGS Publication ID (assigned by PGS Catalog after submission)")
    p.add_argument("--method_name", default="LDpred2-auto")
    p.add_argument("--method_params",
                   default=("vec_p_init=seq_log(1e-4,0.9,length.out=ncores); "
                            "h2_init=ldsc; chains=ncores; "
                            "QC: |pred_scaled-median|<3*MAD; final_beta=rowMeans(retained_chains$beta_est)"))
    p.add_argument("--trait_reported_prefix",
                   default="Cardiac MRI imaging-derived latent feature ",
                   help="Prefix prepended to the score name for trait_reported")
    p.add_argument("--trait_efo", default="",
                   help="trait_efo header value (e.g. 'EFO:0022611').")
    p.add_argument("--trait_mapped", default="",
                   help="trait_mapped header value (EFO term name).")
    p.add_argument("--name_strip_suffix", default="",
                   help=("Suffix to strip from the input filename stem to get "
                         "pgs_name (e.g. '_combined' for disease-level files)."))
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.scoring.raw.tsv"))
    if not raw_files:
        print(f"[ERROR] no *.scoring.raw.tsv files in {raw_dir}", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    summary = []
    for raw in raw_files:
        stem = raw.name.replace(".scoring.raw.tsv", "")
        if args.name_strip_suffix and stem.endswith(args.name_strip_suffix):
            pgs_name = stem[: -len(args.name_strip_suffix)]
        else:
            pgs_name = stem
        out_path = out_dir / f"{pgs_name}.txt.gz"
        meta = {
            "format_version": "2.0",
            "pgs_id": "",  # assigned post-submission
            "pgs_name": pgs_name,
            "trait_reported": f"{args.trait_reported_prefix}{pgs_name}",
            "trait_mapped": args.trait_mapped,
            "trait_efo": args.trait_efo,
            "genome_build": args.genome_build,
            "variants_number": 0,  # filled by wrap_one
            "weight_type": args.weight_type,
            "pgp_id": args.pgp_id,
            "citation": args.citation,
            # No HmPOS_*: these files are NOT harmonised. PGS Catalog
            # curators publish a harmonised copy after acceptance.
            "license": args.license,
            "pgs_method_name": args.method_name,
            "pgs_method_params": args.method_params,
        }
        _ = today  # noqa: F841 (kept for future opt-in HmPOS_date support)
        nvar = wrap_one(raw, out_path, meta)
        print(f"[OK] {pgs_name}: {nvar} variants -> {out_path}")
        summary.append((pgs_name, nvar, str(out_path)))

    sum_path = out_dir / "_wrap_summary.tsv"
    pd.DataFrame(summary, columns=["pgs_name", "n_variants", "path"]).to_csv(
        sum_path, sep="\t", index=False
    )
    print(f"[OK] summary -> {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
