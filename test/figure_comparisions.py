"""
Show how H1/H2/H4 error (vspy - VS reference) varies with F0.

Errors are largest at F0 extremes (vocal fry / low F0, and
child-/falsetto-range / high F0), and smallest in the normal adult
speech range (~120-200 Hz). This script visualizes that relationship.

Usage:
    python test/plot_harmonic_f0_errors.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["H1", "H2", "H4"]

F0_BINS = list(range(0, 501, 50))  # 50 Hz buckets, 0-500 Hz (covers observed F0 range)


def main():
    out = pd.read_csv("test/output.csv")
    errs = pd.read_csv("test/compared_results/per_frame_errors.csv")

    out["_stem"] = out["filename"].str.rsplit(".", n=1).str[0]
    merged = pd.merge(errs, out[["_stem", "t_ms", "f0_snack"]], on=["_stem", "t_ms"])

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(15, 5), sharey=True)

    for ax, feat in zip(axes, FEATURES):
        col = f"err_{feat}"
        sub = merged[["f0_snack", col]].dropna()

        ax.scatter(sub["f0_snack"], sub[col], s=2, alpha=0.15, color="steelblue")

        # binned mean absolute error, overlaid
        sub = sub.copy()
        sub["f0_bin"] = pd.cut(sub["f0_snack"], bins=F0_BINS)
        binned = sub.groupby("f0_bin", observed=True)[col].apply(lambda x: x.abs().mean())
        centers = [iv.mid for iv in binned.index]
        ax.plot(centers, binned.values, color="darkorange", marker="o", label="mean |error|")
        ax.plot(centers, -binned.values, color="darkorange", linestyle="--", alpha=0.5)

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("F0_snack (Hz)")
        ax.set_title(f"{feat}: vspy - VS reference")
        ax.legend()

    axes[0].set_ylabel("error (dB)")
    fig.suptitle("Harmonic amplitude error vs. F0")
    plt.tight_layout()

    out_path = "test/figures/harmonic_error_vs_f0.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")

    # --- simpler summary: mean |error| per F0 bin, as a bar chart ---
    bin_labels = [f"{lo}-{hi}" for lo, hi in zip(F0_BINS[:-1], F0_BINS[1:])]
    table = {}
    for feat in FEATURES:
        col = f"err_{feat}"
        sub = merged[["f0_snack", col]].dropna().copy()
        sub["f0_bin"] = pd.cut(sub["f0_snack"], bins=F0_BINS, labels=bin_labels)
        table[feat] = sub.groupby("f0_bin", observed=True)[col].apply(lambda x: x.abs().mean())

    summary = pd.DataFrame(table)
    print("\nmean |error| by F0 bin:")
    print(summary)

    out_csv = "test/figures/harmonic_error_by_f0_bin.csv"
    summary.to_csv(out_csv)
    print(f"saved -> {out_csv}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bin_labels))
    width = 0.25
    for i, feat in enumerate(FEATURES):
        ax2.bar(x + (i - 1) * width, summary[feat], width, label=feat)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.set_xlabel("F0_snack range (Hz)")
    ax2.set_ylabel("mean |error| (dB)")
    ax2.set_title("Mean |H1/H2/H4 error| by F0 range")
    ax2.legend()
    plt.tight_layout()

    out_path2 = "test/figures/harmonic_error_by_f0_bin.png"
    plt.savefig(out_path2, dpi=150)
    print(f"saved -> {out_path2}")


DIST_BINS = [0, 20, 50, 100, 200, 500, 1200]  # max observed dist(2*F0 or 4*F0, F1/F2) ~1167 Hz
MIN_BIN_N = 10  # drop dist bins with fewer than this many frames (likely tracking errors, not signal)


def plot_correction_vs_resonance_distance():
    """Show that correction (H2c/H4c) adds more error than it removes when
    the harmonic frequency (2*F0 / 4*F0) lands close to a nearby formant
    (a nearby resonance also makes the underlying harmonic amplitude itself
    harder to pin down accurately)."""
    out = pd.read_csv("test/output.csv")
    errs = pd.read_csv("test/compared_results/per_frame_errors.csv")

    out["_stem"] = out["filename"].str.rsplit(".", n=1).str[0]
    merged = pd.merge(
        errs, out[["_stem", "t_ms", "f0_snack", "F1_snack", "F2_snack"]],
        on=["_stem", "t_ms"],
    )

    # (uncorrected col, corrected col, harmonic multiple, formant cols to check proximity to)
    triples = [
        ("H2", "H2c", 2, ["F1_snack", "F2_snack"]),
        ("H4", "H4c", 4, ["F1_snack", "F2_snack"]),
    ]
    bin_labels = [f"{lo}-{hi}" for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]

    fig, axes = plt.subplots(1, len(triples), figsize=(15, 5), sharey=True)

    for ax, (uncorr, corr, mult, formant_cols) in zip(axes, triples):
        cols = ["f0_snack", *formant_cols, f"err_{uncorr}", f"err_{corr}"]
        m = merged[cols].dropna().copy()

        target = mult * m["f0_snack"]
        m["dist"] = np.min(
            np.abs(np.column_stack([target - m[c] for c in formant_cols])), axis=1
        )
        m["delta"] = m[f"err_{corr}"].abs() - m[f"err_{uncorr}"].abs()

        m["dist_bin"] = pd.cut(m["dist"], bins=DIST_BINS, labels=bin_labels)
        grp = m.groupby("dist_bin", observed=True)["delta"]
        binned = grp.mean()
        counts = grp.size()
        binned = binned.where(counts.reindex(binned.index) >= MIN_BIN_N)
        small_bins = counts[counts < MIN_BIN_N]
        if len(small_bins):
            print(f"  {corr}: dropping bins with n<{MIN_BIN_N}: "
                  + ", ".join(f"{b} (n={n})" for b, n in small_bins.items()))

        ax.bar(bin_labels, binned.reindex(bin_labels), color="indianred")
        ax.axhline(0, color="gray", linewidth=0.8)
        formant_label = "/".join(c.split("_")[0] for c in formant_cols)
        ax.set_xlabel(f"distance({mult}*F0, nearest of {formant_label}) [Hz]")
        ax.set_title(f"{corr}: extra error from correction\n(vs. {formant_label})")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("mean(|err_corrected| - |err_uncorrected|)  [dB]")
    fig.suptitle("Correction damage grows as the harmonic nears a formant resonance")
    plt.tight_layout()

    out_path = "test/figures/correction_vs_resonance_distance.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


def plot_correction_vs_resonance_distance_normalized():
    """Same as plot_correction_vs_resonance_distance, but z-scores err and
    err_corrected within each F0 bin first. This removes the F0-extreme
    U-shape (which inflates both uncorrected and corrected error together)
    so the resulting delta isolates the effect of formant-proximity on
    correction specifically."""
    out = pd.read_csv("test/output.csv")
    errs = pd.read_csv("test/compared_results/per_frame_errors.csv")

    out["_stem"] = out["filename"].str.rsplit(".", n=1).str[0]
    merged = pd.merge(
        errs, out[["_stem", "t_ms", "f0_snack", "F1_snack", "F2_snack"]],
        on=["_stem", "t_ms"],
    )

    triples = [
        ("H2", "H2c", 2, ["F1_snack", "F2_snack"]),
        ("H4", "H4c", 4, ["F1_snack", "F2_snack"]),
    ]
    bin_labels = [f"{lo}-{hi}" for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]

    fig, axes = plt.subplots(1, len(triples), figsize=(11, 5), sharey=True)

    for ax, (uncorr, corr, mult, formant_cols) in zip(axes, triples):
        cols = ["f0_snack", *formant_cols, f"err_{uncorr}", f"err_{corr}"]
        m = merged[cols].dropna().copy()

        # z-score |err| and |err_corrected| within each F0 bin, so a frame's
        # error is expressed relative to what's typical for its own F0 range
        m["f0_bin_z"] = pd.cut(m["f0_snack"], bins=F0_BINS)
        for col in [f"err_{uncorr}", f"err_{corr}"]:
            abs_col = m[col].abs()
            grp = abs_col.groupby(m["f0_bin_z"])
            m[f"z_{col}"] = (abs_col - grp.transform("mean")) / grp.transform("std")

        m["delta_z"] = m[f"z_err_{corr}"] - m[f"z_err_{uncorr}"]

        target = mult * m["f0_snack"]
        m["dist"] = np.min(
            np.abs(np.column_stack([target - m[c] for c in formant_cols])), axis=1
        )

        m = m.dropna(subset=["delta_z"])
        m["dist_bin"] = pd.cut(m["dist"], bins=DIST_BINS, labels=bin_labels)
        grp = m.groupby("dist_bin", observed=True)["delta_z"]
        binned = grp.mean()
        counts = grp.size()
        binned = binned.where(counts.reindex(binned.index) >= MIN_BIN_N)
        small_bins = counts[counts < MIN_BIN_N]
        if len(small_bins):
            print(f"  {corr}: dropping bins with n<{MIN_BIN_N}: "
                  + ", ".join(f"{b} (n={n})" for b, n in small_bins.items()))

        ax.bar(bin_labels, binned.reindex(bin_labels), color="seagreen")
        ax.axhline(0, color="gray", linewidth=0.8)
        formant_label = "/".join(c.split("_")[0] for c in formant_cols)
        ax.set_xlabel(f"distance({mult}*F0, nearest of {formant_label}) [Hz]")
        ax.set_title(f"{corr}: extra error from correction\n(F0-normalized)")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("mean(z(|err_corrected|) - z(|err_uncorrected|))")
    fig.suptitle("Correction damage vs. formant proximity, controlling for F0-extreme noise")
    plt.tight_layout()

    out_path = "test/figures/correction_vs_resonance_distance_normalized.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
    plot_correction_vs_resonance_distance()
    plot_correction_vs_resonance_distance_normalized()
