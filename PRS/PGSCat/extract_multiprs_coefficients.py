#!/usr/bin/env python
"""extract_multiprs_coefficients.py

Walk a directory of multi-PRS evaluation results (``*_raw_results.pkl``
produced by ``PRS/eval_multiPRS_V2_wCovar.py``) and extract ONLY the
trained Lasso ``multiPRSCovar`` coefficients (Feature, Coefficient).

This is the safe-to-share subset of the multi-PRS results: it contains
only model weights, no per-individual data.

Output: one TSV per disease in ``--out_dir``, columns:
    disease, fold, feature, coefficient

Note: the intercept is not stored in the source pickle (``multiPRS_predictPanCohort_V2.py``
never saves ``lasso_model.intercept_`` to disk), so it cannot be recovered.

A combined long-format TSV is also written.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


# Some pickles were dumped from __main__ with this helper defined; make
# sure the unpickler can find it.
sys.modules['__main__'].recursive_defaultdict = recursive_defaultdict


PAPER_MODEL = ("Lasso", "multiPRSCovar")


def _find_coeff(obj):
    """Return the Coeff DataFrame from a multiPRSCovar dict.

    The original pipeline stores ``Coeff`` (DataFrame with
    ``Feature, Coefficient``).  The intercept is intentionally not stored
    by ``multiPRS_predictPanCohort_V2.py``, so it is not extracted here.
    """
    if obj is None or not hasattr(obj, "get"):
        return None
    return obj.get("Coeff")


def _walk_folds(res, disease):
    """Yield (fold_id, Coeff_df) for the paper's Lasso multiPRSCovar."""
    if not hasattr(res, "items"):
        return
    for fold_id, fold_res in res.items():
        if not hasattr(fold_res, "get"):
            continue
        model = fold_res.get(PAPER_MODEL[0])
        if model is None or not hasattr(model, "get"):
            continue
        mpc = model.get(PAPER_MODEL[1])
        coeff = _find_coeff(mpc)
        if coeff is not None:
            yield fold_id, coeff


def process_one(pkl_path: Path, out_dir: Path):
    disease = pkl_path.name.replace("_raw_results.pkl", "")
    with open(pkl_path, "rb") as fh:
        res = pickle.load(fh)

    rows = []
    for fold_id, coeff_df in _walk_folds(res, disease):
        if not isinstance(coeff_df, pd.DataFrame):
            try:
                coeff_df = pd.DataFrame(coeff_df)
            except Exception:
                continue
        feat_col = "Feature" if "Feature" in coeff_df.columns else coeff_df.columns[0]
        coef_col = "Coefficient" if "Coefficient" in coeff_df.columns else coeff_df.columns[1]
        for _, row in coeff_df.iterrows():
            rows.append({
                "disease": disease,
                "fold": fold_id,
                "feature": row[feat_col],
                "coefficient": float(row[coef_col]) if not pd.isna(row[coef_col]) else np.nan,
            })

    if not rows:
        print(f"[WARN] {disease}: no Lasso multiPRSCovar coefficients found")
        return None

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{disease}_multiPRS_Lasso_coefficients.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] {disease}: {len(df)} rows -> {out_path}")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True,
                   help="Directory containing *_raw_results.pkl files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pattern", default="*_raw_results.pkl",
                   help="Glob pattern (default: *_raw_results.pkl)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(Path(args.results_dir).glob(args.pattern))
    if not pkls:
        print(f"[ERROR] no pkls in {args.results_dir} matching {args.pattern}",
              file=sys.stderr)
        return 1

    combined = []
    for pkl in pkls:
        try:
            df = process_one(pkl, out_dir)
            if df is not None:
                combined.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] failed for {pkl.name}: {exc}", file=sys.stderr)

    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        all_path = out_dir / "all_multiPRS_Lasso_coefficients.tsv"
        all_df.to_csv(all_path, sep="\t", index=False)
        print(f"[OK] combined -> {all_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
