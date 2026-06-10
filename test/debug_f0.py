"""
Compare vspy Snack F0 against VoiceSauce Snack F0 at voicing boundaries.

Usage:
    python test/debug_f0.py alice_b
    python test/debug_f0.py alice_b --n 20
    python test/debug_f0.py alice_b --type fa   # false alarms only (vspy voiced, VS unvoiced)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from vspy.features.get_pitch_snack import get_pitch_snack, _local_costs


VSPY_CSV  = Path("test/output.csv")
VS_CSV    = Path("test/output_vs.csv")
WAV_DIR   = Path("test/wav")


def load_vs_f0(stem):
    df = pd.read_csv(VS_CSV)
    df["_stem"] = df["Filename"].str.rsplit(".", n=1).str[0]
    sub = df[df["_stem"] == stem][["t_ms", "sF0"]].copy()
    sub["sF0"] = pd.to_numeric(sub["sF0"], errors="coerce").fillna(0.0)
    return sub.set_index("t_ms")["sF0"]


def find_boundary_frames(f0_vspy, vs_f0, kind):
    """
    kind: 'miss'  → vspy unvoiced, VS voiced
          'fa'    → vspy voiced,   VS unvoiced
          'both'  → either
    """
    results = []
    for t_ms in vs_f0.index:
        idx = round(t_ms)
        if idx < 0 or idx >= len(f0_vspy):
            continue
        vspy_val = f0_vspy[idx]
        vs_val   = float(vs_f0.loc[t_ms])

        vspy_voiced = np.isfinite(vspy_val) and vspy_val > 0
        vs_voiced   = vs_val > 0

        if kind in ("miss", "both") and (not vspy_voiced) and vs_voiced:
            results.append((idx, t_ms, "MISS", vspy_val, vs_val))
        elif kind in ("fa", "both") and vspy_voiced and (not vs_voiced):
            results.append((idx, t_ms, "FA",   vspy_val, vs_val))

    return results


def print_frame(frame_idx, t_ms, kind, vspy_val, vs_val, dbg):
    cands    = dbg["all_candidates"][frame_idx]
    max_val  = dbg["all_max_vals"][frame_idx]
    coarse   = dbg["all_coarse"][frame_idx]
    k_max    = dbg["k_max"]
    fs       = dbg["fs"]
    vbias    = dbg["voice_bias"]
    lag_wt   = dbg["lag_weight"]
    stat     = dbg["stats"][frame_idx]
    rms_r    = dbg["rms_ratios"][frame_idx]
    cum      = dbg["cum_costs"][frame_idx]

    local    = _local_costs(cands, max_val, k_max, lag_wt, vbias)
    n_voiced = len(cands)

    nccf_ds  = coarse["nccf_ds"]
    k_min_ds = coarse["k_min_ds"]
    Fds      = coarse["Fds"]
    dec      = coarse["dec"]
    coarse_max = nccf_ds.max() if len(nccf_ds) else 0.0
    coarse_thresh = 0.3 * coarse_max

    tag = "vspy=UNVOICED  VS=voiced" if kind == "MISS" else "vspy=voiced   VS=UNVOICED"
    print(f"\n{'='*60}")
    print(f"frame {frame_idx:4d}  t={t_ms:.0f}ms  [{tag}]")
    print(f"  VS F0={vs_val:.1f} Hz   vspy F0={'NaN' if not np.isfinite(vspy_val) else f'{vspy_val:.1f} Hz'}")
    print(f"  stat={stat:.4f}  rms_ratio={rms_r:.4f}")

    # --- coarse NCCF at the VS-reported lag ---
    if vs_val > 0:
        lag_ds_expected = Fds / vs_val
        ki_expected     = round(lag_ds_expected) - k_min_ds
        if 0 <= ki_expected < len(nccf_ds):
            nccf_at_vs = nccf_ds[ki_expected]
            note = "ABOVE thresh" if nccf_at_vs > coarse_thresh else "BELOW thresh"
            print(f"\n  Coarse NCCF at VS F0 ({vs_val:.1f} Hz → lag_ds≈{lag_ds_expected:.1f}):")
            print(f"    nccf_ds = {nccf_at_vs:.4f}  |  threshold = {coarse_thresh:.4f} ({note})")
        else:
            print(f"\n  Coarse: lag_ds≈{lag_ds_expected:.1f} is outside computed range [{k_min_ds}, {k_min_ds+len(nccf_ds)-1}]")

    # --- all coarse candidates that passed ---
    print(f"\n  Coarse candidates (raw, before lag weight):  max={coarse_max:.4f}  thresh={coarse_thresh:.4f}")
    if coarse["cands_raw"]:
        for lag_ds, peak_ds in coarse["cands_raw"]:
            print(f"    lag_ds={lag_ds:3.1f}  f0_ds={Fds/lag_ds:6.1f}Hz  nccf={peak_ds:.4f}  → full lag≈{round(lag_ds*dec)}")
    else:
        print("    (none)")

    # --- fine candidates and DP ---
    print(f"\n  Fine candidates ({n_voiced} voiced + 1 unvoiced):  max_val={max_val:.4f}")
    for ci, (lag, peak, *_) in enumerate(cands):
        f0_hz = fs / lag if lag > 0 else 0
        print(f"    [{ci}] lag={lag:4d}  f0={f0_hz:6.1f}Hz  peak={peak:.4f}  local={local[ci]:.4f}  dp_cum={cum[ci]:.4f}")
    print(f"    [U] unvoiced  local={local[-1]:.4f}  dp_cum={cum[-1]:.4f}")

    winner = int(np.argmin(cum))
    if winner < n_voiced:
        lag_w = cands[winner][0]
        print(f"\n  DP winner: [{winner}]  f0={fs/lag_w:.1f}Hz  cum={cum[winner]:.4f}")
    else:
        print(f"\n  DP winner: UNVOICED  cum={cum[-1]:.4f}")
        if cands:
            best_voiced = int(np.argmin(cum[:n_voiced]))
            print(f"  Best voiced: [{best_voiced}]  cum={cum[best_voiced]:.4f}  "
                  f"margin (unvoiced-voiced)={cum[-1] - cum[best_voiced]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem",          help="file stem, e.g. alice_b")
    ap.add_argument("--n",    type=int, default=15,
                    help="number of boundary frames to show (default 15)")
    ap.add_argument("--type", choices=["miss","fa","both"], default="miss",
                    help="miss=vspy unvoiced VS voiced, fa=vspy voiced VS unvoiced")
    args = ap.parse_args()

    wav = WAV_DIR / f"{args.stem}.wav"
    if not wav.exists():
        sys.exit(f"WAV not found: {wav}")

    print(f"Running get_pitch_snack on {wav} ...")
    f0, dbg = get_pitch_snack(str(wav), _debug=True)

    print(f"Loading VoiceSauce reference for {args.stem} ...")
    vs_f0 = load_vs_f0(args.stem)

    boundaries = find_boundary_frames(f0, vs_f0, args.type)
    print(f"\nFound {len(boundaries)} {args.type} frames. Showing first {args.n}.\n")

    for frame_idx, t_ms, kind, vspy_val, vs_val in boundaries[:args.n]:
        print_frame(frame_idx, t_ms, kind, vspy_val, vs_val, dbg)

    # summary
    n_miss = sum(1 for *_, k, __, ___ in [(b[0],b[1],b[2],b[3],b[4]) for b in boundaries] if k=="MISS")
    n_fa   = len(boundaries) - n_miss
    total  = len(vs_f0)
    print(f"\n{'='*60}")
    print(f"Summary for {args.stem}:")
    print(f"  total frames : {total}")
    print(f"  misses       : {n_miss}  ({100*n_miss/total:.1f}%)  vspy unvoiced, VS voiced")
    print(f"  false alarms : {n_fa}   ({100*n_fa/total:.1f}%)  vspy voiced, VS unvoiced")


if __name__ == "__main__":
    main()
