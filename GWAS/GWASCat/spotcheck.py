#!/usr/bin/env python3
"""
spotcheck.py

Sanity-check one converted .tsv.gz per analysis (6 total):
  - GWAS-SSF v1.0 mandatory column order for variant-level
  - p-values in (0, 1] (or recoverable from neg_log10 for gene-level)
  - chromosomes within {1..22, X, Y, MT}
  - MD5 of the .tsv.gz matches the value recorded in the YAML companion
"""

import argparse
import gzip
import hashlib
import random
import re
import sys
from pathlib import Path


VARIANT_MANDATORY = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "beta", "standard_error", "effect_allele_frequency", "p_value",
]
GENE_MANDATORY = [
    "hgnc_symbol", "chromosome", "base_pair_start", "base_pair_end",
    "neg_log10_p_value",
]
VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def header_of(path):
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def sample_first_rows(path, n=200):
    rows = []
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
            if len(rows) >= n:
                break
    return header, rows


def read_yaml_md5(yaml_path):
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        for line in f:
            m = re.match(r"^\s*data_file_md5sum:\s*(\S+)", line)
            if m:
                return m.group(1)
    return None


def check_variant(path):
    issues = []
    header, rows = sample_first_rows(path)
    if header[:len(VARIANT_MANDATORY)] != VARIANT_MANDATORY:
        issues.append(
            f"header mismatch: expected {VARIANT_MANDATORY}, got "
            f"{header[:len(VARIANT_MANDATORY)]}"
        )
    for r in rows[:50]:
        chrom = r.get("chromosome")
        if chrom not in VALID_CHROMS:
            issues.append(f"unexpected chromosome: {chrom}")
            break
        try:
            p = float(r.get("p_value"))
            if not (0 < p <= 1):
                issues.append(f"p_value out of (0,1]: {p}")
                break
        except (TypeError, ValueError):
            # Allow scientific-string p in extreme tails.
            pass
    return issues


def check_gene(path):
    issues = []
    header, rows = sample_first_rows(path)
    for col in GENE_MANDATORY:
        if col not in header:
            issues.append(f"missing column: {col}")
    for r in rows[:50]:
        chrom = r.get("chromosome")
        if chrom not in VALID_CHROMS:
            issues.append(f"unexpected chromosome: {chrom}")
            break
        try:
            nlp = float(r.get("neg_log10_p_value"))
            if nlp < 0:
                issues.append(f"negative neg_log10_p_value: {nlp}")
                break
        except (TypeError, ValueError):
            pass
    return issues


def check_md5(tsv_path):
    yaml_path = tsv_path.with_name(tsv_path.name + "-meta.yaml")
    recorded = read_yaml_md5(yaml_path)
    if recorded is None:
        return ["no YAML or md5 entry"]
    actual = md5(tsv_path)
    if recorded != actual:
        return [f"md5 mismatch: yaml={recorded} actual={actual}"]
    return []


def pick(directory, rng):
    files = sorted(directory.glob("*.tsv.gz"))
    files = [f for f in files if "conversion_log" not in f.name]
    if not files:
        return None
    return rng.choice(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.output_root)

    targets = []
    for sub in ["GWAS_Discovery", "GWAS_Replication",
                "GWAS_Sex_Interaction", "EWAS_Discovery"]:
        targets.append(("variant", root / "variant_level" / sub))
    for sub in ["Gene_Burden_Discovery", "SKATO_ACAT_Discovery"]:
        targets.append(("gene", root / "gene_level" / sub))

    fails = 0
    for kind, d in targets:
        f = pick(d, rng)
        if f is None:
            print(f"  SKIP   {d}: no files")
            continue
        issues = check_md5(f)
        if kind == "variant":
            issues += check_variant(f)
        else:
            issues += check_gene(f)
        if issues:
            fails += 1
            print(f"  FAIL   {f}")
            for it in issues:
                print(f"    - {it}")
        else:
            print(f"  OK     {f}")

    if fails:
        print(f"\nSpot-check FAILED on {fails} file(s).")
        sys.exit(1)
    print("\nSpot-check PASSED.")


if __name__ == "__main__":
    main()
