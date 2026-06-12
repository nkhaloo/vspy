"""Stage-by-stage diff of the Python Snack port vs the real Snack DSP routines.

Reproduces the eval's exact int16 input (resample->16k->PCM_16->read back->quantize),
dumps the Python downsample/highpass/poles, runs the C harness on the same input,
and reports the first stage that diverges.
"""
import subprocess, tempfile, math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import vspy.features.get_formants_snack as g
from vspy.io import read_wav

HERE = Path(__file__).parent
DUMP = HERE / "dump"; DUMP.mkdir(exist_ok=True)
LPC_ORD = 12

def make_input_i16(wav):
    y, fs = read_wav(wav)
    if fs != 16000:
        gg = math.gcd(int(fs), 16000)
        y = resample_poly(y, 16000 // gg, int(fs) // gg); fs = 16000
    # round-trip through PCM_16 temp wav exactly like api.vspy()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tp = Path(tmp.name)
    sf.write(tp, y, fs, subtype="PCM_16")
    yb, fsb = read_wav(tp); tp.unlink()
    sig = np.asarray(yb, dtype=np.float64)
    if np.abs(sig).max() <= 1.0:
        sig = sig * 32767.0
    sig = sig.clip(-32768, 32767).round().astype(np.int16)
    return sig, fsb

def py_stages(sig_i16, fs):
    # downsample
    insert, decimate = g._ratprx(10000.0 / fs)
    ds, ds_rate = g._downsample_snack(sig_i16, fs, insert, decimate)
    # highpass (same code preprocess uses)
    LCSIZ = 101; ncoef = 1 + LCSIZ // 2
    fn = np.pi * 2.0 / (LCSIZ - 1); scale = 32767.0 / (0.5 * LCSIZ)
    lcf = (scale * (0.5 + 0.4 * np.cos(fn * np.arange(ncoef)))).astype(np.int16)
    hp = g._do_fir(ds, lcf, ncoef, invert=True)
    # poles
    poles = g.lpc_poles(hp, int(ds_rate), 0.001, LPC_ORD)
    return ds, hp, ds_rate, poles

def load_snack_poles(path):
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        rms = float(p[0]); nf = int(p[1])
        vals = list(map(float, p[2:]))
        freq = np.array(vals[0::2]); band = np.array(vals[1::2])
        out.append({"rms": rms, "nform": nf, "freq": freq, "band": band})
    return out

def main():
    wav = HERE.parent / "wav" / "alice_b.wav"
    sig_i16, fs = make_input_i16(wav)
    (DUMP / "in.raw").write_bytes(sig_i16.tobytes())

    ds, hp, ds_rate, mypoles = py_stages(sig_i16, fs)

    import os
    env = dict(os.environ, DYLD_LIBRARY_PATH=str((HERE.parent.parent / "VoiceSauce" / "Mac").resolve()))
    subprocess.run([str(HERE / "snack_dump"),
                    str(DUMP / "in.raw"), str(fs), str(LPC_ORD), str(DUMP)],
                   check=True, env=env)

    sds = np.fromfile(DUMP / "ds.raw", dtype=np.int16)
    shp = np.fromfile(DUMP / "hp.raw", dtype=np.int16)
    spoles = load_snack_poles(DUMP / "poles.txt")

    print(f"\ninput samples: {len(sig_i16)}  fs={fs}  ds_rate={ds_rate}")
    print(f"--- DOWNSAMPLE ---  mine={len(ds)} snack={len(sds)}")
    n = min(len(ds), len(sds))
    d = ds[:n].astype(int) - sds[:n].astype(int)
    print(f"   exact-equal: {np.mean(d==0)*100:.2f}%   max|d|={np.abs(d).max()}   mean|d|={np.abs(d).mean():.3f}")
    print(f"--- HIGHPASS ---  mine={len(hp)} snack={len(shp)}")
    n = min(len(hp), len(shp))
    d = hp[:n].astype(int) - shp[:n].astype(int)
    print(f"   exact-equal: {np.mean(d==0)*100:.2f}%   max|d|={np.abs(d).max()}   mean|d|={np.abs(d).mean():.3f}")

    print(f"--- POLES ---  mine={len(mypoles)} snack={len(spoles)} frames")
    nfr = min(len(mypoles), len(spoles))
    fdiffs, ndiff = [], 0
    for i in range(nfr):
        mf = np.sort(mypoles[i]["freq"]); sf_ = np.sort(spoles[i]["freq"])
        if len(mf) != len(sf_):
            ndiff += 1; continue
        if len(mf):
            fdiffs.append(np.abs(mf - sf_).max())
    fdiffs = np.array(fdiffs) if fdiffs else np.array([0.0])
    print(f"   frames w/ different #poles: {ndiff}/{nfr}")
    print(f"   per-frame max pole-freq diff (matched-count frames): "
          f"mean={fdiffs.mean():.2f}  median={np.median(fdiffs):.2f}  p95={np.percentile(fdiffs,95):.1f}  max={fdiffs.max():.0f}")
    # show a few frames
    print("\n   sample frames (snack vs mine, sorted freqs):")
    shown = 0
    for i in range(nfr):
        if spoles[i]["nform"] == 0: continue
        sf_ = np.sort(spoles[i]["freq"]); mf = np.sort(mypoles[i]["freq"])
        print(f"   fr{i}: snack {np.round(sf_,0)}  | mine {np.round(mf,0)}")
        shown += 1
        if shown >= 8: break

if __name__ == "__main__":
    main()
