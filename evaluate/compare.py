"""Compare one corpus. Run python -m evaluate.compare evaluate/buckeye from the repo root."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .extract import FEATURE_PAIRS, F0_PAIRS, find_wavs

# Metric definitions and rounding follow test/evaluate.py.
MAD_K = 3.5


def mad_outlier_mask(x, k=MAD_K):
    """Modified z-score (Iglewicz & Hoaglin) outlier mask via MAD.

    Returns a boolean array, True where |0.6745*(x-median)/MAD| > k.
    If MAD == 0 (no spread), nothing is flagged.
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return np.zeros(len(x), dtype=bool)
    modified_z = 0.6745 * (x - median) / mad
    return np.abs(modified_z) > k


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
    """Core metrics for a matched pair of arrays. total_rows is the merged row count.

    Frames are dropped if either side is non-finite, OR if either side's value
    is a MAD-based outlier (modified z-score > MAD_K) computed independently
    per system (vspy vs. VS reference).
    """
    mask = np.isfinite(vspy_vals) & np.isfinite(vs_vals)
    a, b = vspy_vals[mask], vs_vals[mask]
    n_finite = mask.sum()

    outlier_a = mad_outlier_mask(a)
    outlier_b = mad_outlier_mask(b)
    outliers  = outlier_a | outlier_b
    keep = ~outliers
    a, b = a[keep], b[keep]

    n_valid = keep.sum()
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
        "n_finite":   int(n_finite),
        "n_outliers_dropped": int(outliers.sum()),
        "n_outliers_vspy":    int(outlier_a.sum()),
        "n_outliers_vs":      int(outlier_b.sum()),
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


def split_reference(corpus, wavs, tmp, pairs, keys):
    """Partition the large reference once. A recording can span several chunks."""
    names = {w.stem + ".mat": w.stem for w in wavs}
    measures = [ref for _, ref, _ in pairs]
    columns = ["Filename", *keys, *measures]
    missing_values = {name: ["", "NA", "NaN", "nan"] for name in measures}
    # Use explicit numeric types so all-NA chunks have the same schema.
    types = {name: "float64" for name in [*measures, "t_ms", "seg_Start", "seg_End"]}
    with pd.read_csv(corpus / "output.txt", sep="\t", usecols=columns, dtype=types,
                     keep_default_na=False, na_values=missing_values, chunksize=60_000) as chunks:
        for number, chunk in enumerate(chunks):
            selected = chunk[chunk.Filename.isin(names)]
            for filename, rows in selected.groupby("Filename", sort=False):
                part = tmp / names[filename]
                part.mkdir(exist_ok=True)
                rows.drop(columns="Filename").to_parquet(part / f"{number:08}.parquet", index=False)


def prepare_pairs(corpus, wav, tmp, pairs, keys):
    """Match saved measurements back to reference rows without multiplying repeated labels."""
    prediction = pd.read_parquet(corpus / (wav.stem + ".parquet"))
    parts = sorted((tmp / wav.stem).glob("*.parquet"))
    if not parts or prediction.empty or not prediction.recording_id.eq(wav.stem).all():
        raise ValueError(f"Missing or inconsistent data for {wav.stem}")
    reference = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    expected = reference.loc[reference.t_ms.isin(prediction.t_ms)]
    if len(expected) != len(prediction):
        raise ValueError(f"Unexpected reference alignment for {wav.stem}")
    for table in (prediction, reference):
        table["occurrence"] = table.groupby(keys, dropna=False, sort=False).cumcount()
    left = prediction[[*keys, "occurrence", *[v for v, _, _ in pairs]]]
    right = reference.rename(columns={r: "vs_" + r for _, r, _ in pairs})
    merged = left.merge(right, on=[*keys, "occurrence"], how="left", validate="one_to_one",
                        indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Saved frames do not match the reference in {wav.stem}")
    if "Label" in keys:
        merged = merged.loc[merged.Label != "[SIL]"]
    # Keep signed errors at every evaluated frame; the summary applies the MAD filter.
    errors = merged[keys].copy()
    errors.insert(0, "recording_id", wav.stem)
    for v, r, name in pairs:
        if (v, r, name) in FEATURE_PAIRS:
            errors["err_" + name] = merged[v] - merged["vs_" + r]
    coverage = dict(recording_id=wav.stem, reference_rows=len(reference),
                    first_reference_ms=reference.t_ms.min(), last_reference_ms=reference.t_ms.max(),
                    retained_rows=len(prediction), first_retained_ms=prediction.t_ms.min(),
                    last_retained_ms=prediction.t_ms.max(), evaluated_rows=len(merged))
    return merged, errors, coverage


def compare(corpus):
    out = corpus / "result"
    if out.exists():
        raise FileExistsError(f"Choose a fresh result directory by moving {out} before rerunning")
    wavs = find_wavs(corpus)
    header = pd.read_csv(corpus / "output.txt", sep="\t", nrows=0).columns
    pairs = [p for p in FEATURE_PAIRS + F0_PAIRS if p[1] in header]
    if not pairs:
        raise ValueError("No comparable VoiceSauce measurements")
    keys = ["t_ms", "seg_Start", "seg_End"]
    if "Label" in header:
        keys.append("Label")

    with TemporaryDirectory(dir=corpus) as directory:
        tmp = Path(directory)
        staged = tmp / "result"
        staged.mkdir()
        split_reference(corpus, wavs, tmp, pairs, keys)
        paired = tmp / "paired"
        paired.mkdir()
        coverage_rows = []
        writer = None
        try:
            for wav in wavs:
                frame, errors, coverage = prepare_pairs(corpus, wav, tmp, pairs, keys)
                frame[[c for v, r, _ in pairs for c in (v, "vs_" + r)]].to_parquet(
                    paired / (wav.stem + ".parquet"), index=False)
                batch = pa.Table.from_pandas(errors, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(staged / "per_frame_errors.parquet", batch.schema)
                writer.write_table(batch)
                coverage_rows.append(coverage)
        finally:
            if writer is not None:
                writer.close()

        # Metrics use the pooled frames across all recordings, including a single corpus-wide MAD.
        # Reading one column pair at a time bounds memory without averaging per-file scores.
        general, f0 = [], []
        for v, r, name in pairs:
            table = pd.read_parquet(paired, columns=[v, "vs_" + r])
            a, b = table[v].to_numpy(float), table["vs_" + r].to_numpy(float)
            if (v, r, name) in F0_PAIRS:
                f0.append({"feature": name, **compute_f0_metrics(a, b, len(table))})
            else:
                general.append({"feature": name, **compute_general_metrics(a, b, len(table))})
        pd.DataFrame(general).to_csv(staged / "feature_metrics.csv", index=False)
        pd.DataFrame(f0).to_csv(staged / "f0_metrics.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(staged / "recording_coverage.csv", index=False)
        staged.rename(out)
    print(f"Saved {len(general)} feature and {len(f0)} pitch summaries for {len(wavs)} recordings in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory with output.txt, WAVs, and extracted Parquet files")
    compare(parser.parse_args().corpus)
