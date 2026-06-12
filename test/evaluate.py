"""
Evaluation script: compare vspy output against VoiceSauce reference output.

Usage:
    python test/evaluate.py \
        --vspy   test/output.csv \
        --vs     test/output_vs.csv \
        --outdir test/compared_results
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Column mapping: (vspy_col, voicesauce_col, display_name)
# ---------------------------------------------------------------------------
FEATURE_PAIRS = [
    # formants
    ("F1_praat",  "pF1",    "F1_praat"),
    ("F2_praat",  "pF2",    "F2_praat"),
    ("F3_praat",  "pF3",    "F3_praat"),
    ("F4_praat",  "pF4",    "F4_praat"),
    ("F1_snack",  "sF1",    "F1_snack"),
    ("F2_snack",  "sF2",    "F2_snack"),
    ("F3_snack",  "sF3",    "F3_snack"),
    ("F4_snack",  "sF4",    "F4_snack"),
    # bandwidths
    ("B1_praat",  "pB1",    "B1_praat"),
    ("B2_praat",  "pB2",    "B2_praat"),
    ("B3_praat",  "pB3",    "B3_praat"),
    ("B1_snack",  "sB1",    "B1_snack"),
    ("B2_snack",  "sB2",    "B2_snack"),
    ("B3_snack",  "sB3",    "B3_snack"),
    # uncorrected harmonic amplitudes
    ("H1",        "H1u",    "H1"),
    ("H2",        "H2u",    "H2"),
    ("H4",        "H4u",    "H4"),
    ("A1",        "A1u",    "A1"),
    ("A2",        "A2u",    "A2"),
    ("A3",        "A3u",    "A3"),
    ("H2K",       "H2Ku",   "H2K"),
    ("H5K",       "H5Ku",   "H5K"),
    # corrected harmonic amplitudes
    ("H1c",       "H1c",    "H1c"),
    ("H2c",       "H2c",    "H2c"),
    ("H4c",       "H4c",    "H4c"),
    ("A1c",       "A1c",    "A1c"),
    ("A2c",       "A2c",    "A2c"),
    ("A3c",       "A3c",    "A3c"),
    ("H2Kc",      "H2Kc",   "H2Kc"),
    # spectral tilts
    ("H1H2",      "H1H2c",  "H1H2"),
    ("H2H4",      "H2H4c",  "H2H4"),
    ("H1A1",      "H1A1c",  "H1A1"),
    ("H1A2",      "H1A2c",  "H1A2"),
    ("H1A3",      "H1A3c",  "H1A3"),
    ("H4H2K",     "H42Kc",  "H4H2K"),
    ("H2KH5K",    "H2KH5Kc","H2KH5K"),
]

F0_PAIRS = [
    ("f0_snack", "sF0", "f0_snack"),
    ("f0_praat", "pF0", "f0_praat"),
]

# When --snack-only is passed, restrict the run to just these (snack formants +
# bandwidths) and skip F0. Default is off, so omitting the flag restores the
# full evaluation — nothing else to revert.
SNACK_FB_NAMES = {
    "F1_snack", "F2_snack", "F3_snack", "F4_snack",
    "B1_snack", "B2_snack", "B3_snack",
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def pearson_r(a, b):
    if len(a) < 3:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def ccc(a, b):
    """Concordance Correlation Coefficient (Lin 1989)."""
    if len(a) < 3:
        return np.nan
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    covariance = np.cov(a, b, ddof=0)[0, 1]
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    return 2 * covariance / denom if denom != 0 else np.nan

def compute_general_metrics(vspy_vals, vs_vals, total_rows):
    """Core metrics for a matched pair of arrays. total_rows is the merged row count."""
    mask = np.isfinite(vspy_vals) & np.isfinite(vs_vals)
    n_valid = mask.sum()
    a, b = vspy_vals[mask], vs_vals[mask]
    diff = a - b
    return {
        "pearson_r":  round(pearson_r(a, b), 4) if n_valid >= 3 else np.nan,
        "ccc":        round(ccc(a, b), 4)        if n_valid >= 3 else np.nan,
        "mean_error":   round(diff.mean(), 3)            if n_valid >= 1 else np.nan,
        "median_error": round(np.median(diff), 3)        if n_valid >= 1 else np.nan,
        "mae":          round(np.abs(diff).mean(), 3)    if n_valid >= 1 else np.nan,
        "coverage":   round(n_valid / total_rows, 4) if total_rows > 0 else np.nan,
        "n_valid":    int(n_valid),
        "n_total":    int(total_rows),
    }

def compute_f0_metrics(vspy_f0, vs_f0, total_rows):
    """General metrics plus VDE, GPE, FPE for F0 columns."""
    base = compute_general_metrics(vspy_f0, vs_f0, total_rows)

    voiced_vspy = np.isfinite(vspy_f0) & (vspy_f0 > 0)
    voiced_vs   = np.isfinite(vs_f0)   & (vs_f0   > 0)

    # VDE: frames where voicing decision disagrees
    vde = np.mean(voiced_vspy != voiced_vs) if total_rows > 0 else np.nan

    # agreed-voiced frames
    both_voiced = voiced_vspy & voiced_vs
    n_both = both_voiced.sum()

    if n_both > 0:
        ratio = np.abs(vspy_f0[both_voiced] - vs_f0[both_voiced]) / vs_f0[both_voiced]
        gpe = np.mean(ratio > 0.20)
        fpe_mean = float(np.mean(vspy_f0[both_voiced] - vs_f0[both_voiced]))
        fpe_std  = float(np.std( vspy_f0[both_voiced] - vs_f0[both_voiced]))
    else:
        gpe = fpe_mean = fpe_std = np.nan

    return {
        **base,
        "vde":      round(vde,      4) if not np.isnan(vde)      else np.nan,
        "gpe":      round(gpe,      4) if not np.isnan(gpe)      else np.nan,
        "fpe_mean": round(fpe_mean, 3) if not np.isnan(fpe_mean) else np.nan,
        "fpe_std":  round(fpe_std,  3) if not np.isnan(fpe_std)  else np.nan,
        "n_both_voiced": int(n_both),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(vspy_path, vs_path, outdir, snack_only=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vspy_df = pd.read_csv(vspy_path)
    vs_df   = pd.read_csv(vs_path)

    # normalize filenames to stem for joining
    label_col = "words" if "words" in vspy_df.columns else "phones"
    vspy_df = vspy_df[vspy_df[label_col] != "[SIL]"].copy()
    vspy_df["_stem"] = vspy_df["filename"].str.rsplit(".", n=1).str[0]
    vs_df["_stem"]   = vs_df["Filename"].str.rsplit(".", n=1).str[0]

    # prefix columns before merge to avoid ambiguity on shared names
    vspy_rename = {c: f"v_{c}" for c in vspy_df.columns if c != "_stem" and c != "t_ms"}
    vs_rename   = {c: f"r_{c}" for c in vs_df.columns   if c != "_stem" and c != "t_ms"}
    vspy_df = vspy_df.rename(columns=vspy_rename)
    vs_df   = vs_df.rename(columns=vs_rename)

    merged = pd.merge(vspy_df, vs_df, on=["_stem", "t_ms"])
    total  = len(merged)
    print(f"Merged rows: {total}")

    # --- general features ---
    feature_pairs = FEATURE_PAIRS
    if snack_only:
        feature_pairs = [p for p in FEATURE_PAIRS if p[2] in SNACK_FB_NAMES]

    rows = []
    for vspy_col, vs_col, name in feature_pairs:
        vc = f"v_{vspy_col}"
        rc = f"r_{vs_col}"
        if vc not in merged.columns or rc not in merged.columns:
            print(f"  skipping {name}: column not found ({vc} / {rc})")
            continue
        metrics = compute_general_metrics(
            merged[vc].to_numpy(dtype=float),
            merged[rc].to_numpy(dtype=float),
            total,
        )
        rows.append({"feature": name, **metrics})

    features_df = pd.DataFrame(rows)
    out_features = outdir / "feature_metrics.csv"
    features_df.to_csv(out_features, index=False)
    print(f"Saved feature metrics -> {out_features}")

    # --- F0 ---
    if snack_only:
        print("snack-only: skipping F0 metrics")
        return

    f0_rows = []
    for vspy_col, vs_col, name in F0_PAIRS:
        vc = f"v_{vspy_col}"
        rc = f"r_{vs_col}"
        if vc not in merged.columns or rc not in merged.columns:
            print(f"  skipping {name}: column not found")
            continue
        metrics = compute_f0_metrics(
            merged[vc].to_numpy(dtype=float),
            merged[rc].to_numpy(dtype=float),
            total,
        )
        f0_rows.append({"feature": name, **metrics})

    f0_df = pd.DataFrame(f0_rows)
    out_f0 = outdir / "f0_metrics.csv"
    f0_df.to_csv(out_f0, index=False)
    print(f"Saved F0 metrics     -> {out_f0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vspy",   default="test/output.csv")
    parser.add_argument("--vs",     default="test/output_vs.csv")
    parser.add_argument("--outdir", default="test/compared_results")
    parser.add_argument("--snack-only", action="store_true",
                        help="evaluate only snack formants/bandwidths; skip F0 "
                             "(default off; omit the flag to restore full eval)")
    args = parser.parse_args()
    run(args.vspy, args.vs, args.outdir, snack_only=args.snack_only)
