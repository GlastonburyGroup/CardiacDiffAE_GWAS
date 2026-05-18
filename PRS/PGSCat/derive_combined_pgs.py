#!/usr/bin/env python
"""derive_combined_pgs.py

Back-calculate a single set of SNP-level weights for each disease from
the multi-PRS LASSO results, so the PGS Catalog can host one per-SNP
scoring file per disease (in addition to the 49 per-latent scoring
files).

Maths
-----
The paper's multi-PRS model is

    risk_d  =  intercept_d
             + sum_k  alpha_{d,k} * PRS_k
             + sum_c  gamma_{d,c} * covariate_c

with ``PRS_k = sum_i beta_{k,i} * dosage_i``. Substituting PRS_k:

    risk_d  =  intercept_d
             + sum_i  ( sum_{k: SNP_i in latent_k}
                        alpha_{d,k} * beta_{k,i} ) * dosage_i
             + sum_c  gamma_{d,c} * covariate_c

So the *combined per-SNP weight* for disease ``d`` is simply

    w_d(i)  =  sum_k  alpha_{d,k} * beta_{k,i}

evaluated over all latents ``k`` that contain SNP ``i`` (with the betas
flipped if the latent's effect/other alleles are swapped relative to
the disease scoring file's chosen reference allele).

Important assumptions
---------------------
* The Lasso multiPRSCovar model is fit on **raw-scale** PRS values
  (the paper uses ``multiPRSCovar`` and NOT ``multiPRSNormCovar``).
  If your pickles were produced with the ``Norm`` variant, supply
  per-latent (mean, sd) via ``--prs_scale_tsv`` so the betas can be
  rescaled before combination.
* The covariate terms cannot be folded into per-SNP weights and are
  emitted as a separate ``{disease}_covariate_coefficients.tsv`` sidecar.
  These have to be reported separately by the PGS user when computing
  the full disease-risk score.
* Cross-validation folds are aggregated by ``--fold_agg`` (default:
  mean across folds with a non-zero coefficient for the feature; use
  ``mean_all`` to average across all folds including zeros).
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


# Required so the unpickler can resolve helpers used at dump-time.
sys.modules["__main__"].recursive_defaultdict = recursive_defaultdict


PAPER_MODEL = ("Lasso", "multiPRSCovar")
PRS_PREFIX = "PRS:"


# ------------------------------- helpers -------------------------------

def _walk_folds(res):
    """Yield ``(fold_id, Coeff DataFrame, intercept_or_nan)`` rows.

    Note: ``multiPRS_predictPanCohort_V2.py`` never saves the model intercept
    to the pickle, so the third element will always be ``np.nan``.
    """
    if not hasattr(res, "items"):
        return
    for fold_id, fold_res in res.items():
        if not hasattr(fold_res, "get"):
            continue
        model = fold_res.get(PAPER_MODEL[0])
        if model is None or not hasattr(model, "get"):
            continue
        mpc = model.get(PAPER_MODEL[1])
        if mpc is None or not hasattr(mpc, "get"):
            continue
        coeff = mpc.get("Coeff")
        intercept = mpc.get("Intercept", mpc.get("intercept", np.nan))
        if coeff is None:
            continue
        if not isinstance(coeff, pd.DataFrame):
            try:
                coeff = pd.DataFrame(coeff)
            except Exception:
                continue
        yield fold_id, coeff, intercept


def aggregate_coefficients(coeff_frames, fold_agg: str):
    """Aggregate per-fold ``(Feature, Coefficient)`` frames into one Series.

    ``fold_agg`` ``mean_nz`` averages across folds for which the feature
    had a non-zero coefficient (so a feature retained by Lasso in 2/5
    folds is still represented).  ``mean_all`` averages across all folds
    (counting zeros), and ``median`` returns the per-feature median.
    """
    long = []
    for fold_id, coeff in coeff_frames:
        feat_col = "Feature" if "Feature" in coeff.columns else coeff.columns[0]
        val_col = "Coefficient" if "Coefficient" in coeff.columns else coeff.columns[1]
        tmp = coeff[[feat_col, val_col]].copy()
        tmp.columns = ["feature", "coefficient"]
        tmp["fold"] = fold_id
        long.append(tmp)
    if not long:
        return pd.Series(dtype=float), 0
    long_df = pd.concat(long, ignore_index=True)
    n_folds = long_df["fold"].nunique()
    if fold_agg == "mean_all":
        agg = long_df.groupby("feature")["coefficient"].mean()
    elif fold_agg == "median":
        agg = long_df.groupby("feature")["coefficient"].median()
    else:  # mean_nz (default)
        nz = long_df[long_df["coefficient"].abs() > 0]
        agg = nz.groupby("feature")["coefficient"].sum() / n_folds
        # features always zero across folds become 0:
        zero_feats = set(long_df["feature"]) - set(agg.index)
        if zero_feats:
            agg = pd.concat([agg, pd.Series(0.0, index=list(zero_feats))])
    return agg.sort_index(), n_folds


def split_prs_vs_covar(series: pd.Series):
    prs_mask = series.index.str.startswith(PRS_PREFIX)
    prs = series[prs_mask].copy()
    prs.index = [f[len(PRS_PREFIX):] for f in prs.index]
    covar = series[~prs_mask].copy()
    return prs, covar


def load_latent_betas(raw_dir: Path):
    """Return ``{latent_id: DataFrame[chr, pos, rsID, ea, oa, weight]}``."""
    out = {}
    for raw in sorted(raw_dir.glob("*.scoring.raw.tsv")):
        latent = raw.name.replace(".scoring.raw.tsv", "")
        df = pd.read_csv(raw, sep="\t", dtype={
            "chr_name": str, "chr_position": np.int64,
            "rsID": str, "effect_allele": str,
            "other_allele": str, "effect_weight": float,
        })
        out[latent] = df
    return out


def load_prs_scale(path: Path | None):
    if path is None or not path.exists():
        return None
    scale = pd.read_csv(path, sep="\t")
    if not {"latent", "mean", "sd"}.issubset(scale.columns):
        raise ValueError(f"--prs_scale_tsv must have columns: latent, mean, sd. Got {list(scale.columns)}")
    return scale.set_index("latent")


def combine_for_disease(
    prs_coefs: pd.Series,
    latent_betas: dict[str, pd.DataFrame],
    scale: pd.DataFrame | None,
):
    """Return a per-SNP DataFrame combining LASSO PRS coefs with per-latent betas.

    Output columns: ``chr_name, chr_position, rsID, effect_allele,
    other_allele, effect_weight, n_latents``.
    """
    used = {k: v for k, v in prs_coefs.items() if abs(v) > 0}
    if not used:
        return pd.DataFrame(columns=[
            "chr_name", "chr_position", "rsID", "effect_allele",
            "other_allele", "effect_weight", "n_latents",
        ])

    intercept_offset = 0.0
    snp_acc: dict[tuple, dict] = {}

    for latent, alpha in used.items():
        if latent not in latent_betas:
            print(f"[WARN] LASSO retained {latent} but no scoring file -> skipping",
                  file=sys.stderr)
            continue
        df = latent_betas[latent]
        # Optional re-scaling if model was fit on z-scored PRS:
        eff_alpha = alpha
        if scale is not None and latent in scale.index:
            sd = float(scale.loc[latent, "sd"])
            mu = float(scale.loc[latent, "mean"])
            if sd <= 0:
                raise ValueError(f"sd<=0 for {latent} in --prs_scale_tsv")
            eff_alpha = alpha / sd
            intercept_offset -= alpha * mu / sd

        for chrom, pos, rsid, ea, oa, w in zip(
            df["chr_name"].astype(str),
            df["chr_position"].astype(np.int64),
            df["rsID"].astype(str),
            df["effect_allele"].astype(str),
            df["other_allele"].astype(str),
            df["effect_weight"].astype(float),
        ):
            key = (chrom, int(pos))
            entry = snp_acc.get(key)
            if entry is None:
                snp_acc[key] = {
                    "chr_name": chrom, "chr_position": int(pos),
                    "rsID": rsid, "effect_allele": ea, "other_allele": oa,
                    "effect_weight": eff_alpha * w, "n_latents": 1,
                    "rsID_set": {rsid},
                }
                continue
            # Harmonise alleles: same orientation -> add, flipped -> subtract,
            # otherwise -> skip with a warning (allele mismatch).
            if ea == entry["effect_allele"] and oa == entry["other_allele"]:
                entry["effect_weight"] += eff_alpha * w
            elif ea == entry["other_allele"] and oa == entry["effect_allele"]:
                entry["effect_weight"] -= eff_alpha * w
            else:
                print(f"[WARN] allele mismatch at {chrom}:{pos} between "
                      f"{latent} ({ea}/{oa}) and reference "
                      f"({entry['effect_allele']}/{entry['other_allele']}) "
                      "-> skipping for this latent", file=sys.stderr)
                continue
            entry["n_latents"] += 1
            if rsid and rsid != "NA":
                entry["rsID_set"].add(rsid)

    rows = []
    for entry in snp_acc.values():
        rs_candidates = sorted(x for x in entry["rsID_set"] if x and x != "NA")
        rs_final = rs_candidates[0] if rs_candidates else ""
        rows.append({
            "chr_name": entry["chr_name"],
            "chr_position": entry["chr_position"],
            "rsID": rs_final,
            "effect_allele": entry["effect_allele"],
            "other_allele": entry["other_allele"],
            "effect_weight": entry["effect_weight"],
            "n_latents": entry["n_latents"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["effect_weight"] != 0.0]
    df["chr_position"] = df["chr_position"].astype(np.int64)
    df = df.sort_values(["chr_name", "chr_position"]).reset_index(drop=True)
    df.attrs["intercept_offset"] = intercept_offset
    return df


# ------------------------------- main ---------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True,
                   help="Directory containing *_raw_results.pkl files")
    p.add_argument("--raw_dir", required=True,
                   help="Directory of per-latent *.scoring.raw.tsv files "
                        "(output of extract_scoring_files.R)")
    p.add_argument("--out_dir", required=True,
                   help="Directory for combined per-disease outputs")
    p.add_argument("--pattern", default="*_raw_results.pkl")
    p.add_argument("--fold_agg", choices=("mean_nz", "mean_all", "median"),
                   default="mean_nz",
                   help="How to aggregate LASSO coefficients across CV folds.")
    p.add_argument("--prs_scale_tsv", default=None,
                   help="Optional TSV (latent, mean, sd) for back-rescaling if "
                        "the model was fit on z-scored PRS.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(Path(args.results_dir).glob(args.pattern))
    if not pkls:
        print(f"[ERROR] no pkls in {args.results_dir} matching {args.pattern}",
              file=sys.stderr)
        return 1

    print(f"[INFO] loading per-latent betas from {args.raw_dir} ...")
    latent_betas = load_latent_betas(Path(args.raw_dir))
    print(f"[INFO]   -> {len(latent_betas)} latents loaded")

    scale = load_prs_scale(Path(args.prs_scale_tsv) if args.prs_scale_tsv else None)
    if scale is not None:
        print(f"[INFO] using PRS scaling from {args.prs_scale_tsv}")

    summary_rows = []
    for pkl in pkls:
        disease = pkl.name.replace("_raw_results.pkl", "")
        print(f"\n[{disease}] loading {pkl.name}")
        with open(pkl, "rb") as fh:
            res = pickle.load(fh)
        coeff_frames = [(fid, df) for fid, df, _ in _walk_folds(res)]
        intercepts = [it for _, _, it in _walk_folds(res)
                      if it is not None and not (isinstance(it, float) and np.isnan(it))]
        if not coeff_frames:
            print(f"[WARN] {disease}: no Lasso multiPRSCovar coefficients found")
            continue
        agg, n_folds = aggregate_coefficients(coeff_frames, args.fold_agg)
        prs, covar = split_prs_vs_covar(agg)
        mean_intercept = float(np.mean(intercepts)) if intercepts else float("nan")
        print(f"   folds={n_folds}, retained PRS latents={int((prs.abs() > 0).sum())}, "
              f"retained covariates={int((covar.abs() > 0).sum())}, "
              f"mean intercept={mean_intercept:.4g}")

        combined = combine_for_disease(prs, latent_betas, scale)
        scoring_path = out_dir / f"{disease}_combined.scoring.raw.tsv"
        combined.drop(columns=["n_latents"], errors="ignore").to_csv(
            scoring_path, sep="\t", index=False,
        )
        print(f"   scoring file (raw): {scoring_path} ({len(combined)} variants)")

        # Sidecar with the covariate part (NOT folded into per-SNP weights):
        sidecar_rows = []
        for feat, coef in covar.items():
            sidecar_rows.append({
                "disease": disease,
                "feature": feat,
                "coefficient": float(coef),
                "fold_agg": args.fold_agg,
                "n_folds": n_folds,
            })
        # Intercept is not stored in the source pickle; omit when unavailable.
        if not np.isnan(mean_intercept):
            sidecar_rows.append({
                "disease": disease,
                "feature": "(intercept)",
                "coefficient": mean_intercept,
                "fold_agg": args.fold_agg,
                "n_folds": n_folds,
            })
        intercept_offset = float(combined.attrs.get("intercept_offset", 0.0)) if not combined.empty else 0.0
        if intercept_offset != 0.0:
            sidecar_rows.append({
                "disease": disease,
                "feature": "(intercept_offset_from_PRS_rescaling)",
                "coefficient": intercept_offset,
                "fold_agg": args.fold_agg,
                "n_folds": n_folds,
            })
        sidecar_path = out_dir / f"{disease}_covariate_coefficients.tsv"
        pd.DataFrame(sidecar_rows).to_csv(sidecar_path, sep="\t", index=False)
        print(f"   covariate sidecar: {sidecar_path}")

        summary_rows.append({
            "disease": disease,
            "n_variants_combined": len(combined),
            "n_latents_retained": int((prs.abs() > 0).sum()),
            "n_covariates_retained": int((covar.abs() > 0).sum()),
            "n_folds": n_folds,
            "mean_intercept": mean_intercept,
            "scoring_file": str(scoring_path),
            "covariate_sidecar": str(sidecar_path),
        })

    if summary_rows:
        summary_path = out_dir / "_combined_pgs_summary.tsv"
        pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
        print(f"\n[OK] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
