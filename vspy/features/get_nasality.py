import math
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth import praat

from vspy.align.textgrid import parse_textgrid
from vspy.io import resolve_paths


NASALITY_COLUMNS = (
    "amp_p0",
    "freq_p0",
    "p0_id",
    "p0prominence",
    "a1p0_h1",
    "a1p0_h2",
    "a1p0_h3",
    "a1p0",
    "a1p0_compensated",
    "freq_p1",
    "amp_p1",
    "a1p1",
    "a1p1_Compensated",
    "a3p0",
    "vwl_duration",
)


def _ltas_peak(ltas, low_hz, high_hz):
    """Return Praat's maximum frequency and its nearest-bin level."""
    if not (np.isfinite(low_hz) and np.isfinite(high_hz)) or high_hz <= low_hz:
        return np.nan, np.nan
    try:
        frequency = float(
            praat.call(ltas, "Get frequency of maximum", low_hz, high_hz, "None")
        )
        level = float(praat.call(ltas, "Get value at frequency", frequency, "Nearest"))
        return frequency, level
    except parselmouth.PraatError:
        return np.nan, np.nan


def _chen_term(formant_hz, bandwidth_hz, peak_hz):
    """Formant contribution term transcribed from the Styler Praat script."""
    if not all(np.isfinite(x) for x in (formant_hz, bandwidth_hz, peak_hz)):
        return np.nan
    half_bw_sq = (0.5 * bandwidth_hz) ** 2
    numerator = half_bw_sq + formant_hz**2
    denominator = math.sqrt(
        (half_bw_sq + (formant_hz - peak_hz) ** 2)
        * (half_bw_sq + (formant_hz + peak_hz) ** 2)
    )
    return numerator / denominator if denominator else np.nan


def get_nasality(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    p1search=950.0,
    tpnum=5,
    scoot=0.0,
    intervals=None,
    p0_ceiling_hz=350.0,
    spectrum_window_s=0.030,
):
    """Calculate the selected nasality outputs on the vspy frame grid.

    ``a1p0_h3`` is diagnostic only.  In accordance with Chen (1997) and the
    source script, H3 is never eligible for ``a1p0``/P0 selection.
    """
    sound = parselmouth.Sound(str(wavfile))
    duration_s = sound.get_total_duration()
    if datalen is None:
        datalen = int(duration_s * 1000 / frameshift_ms)
    if tpnum < 1:
        raise ValueError("tpnum must be at least 1")
    if scoot < 0:
        raise ValueError("scoot must be non-negative")
    if p1search <= 100:
        raise ValueError("p1search must be greater than 100 Hz")

    hop_s = frameshift_ms / 1000.0
    pitch = sound.to_pitch_cc(time_step=hop_s, pitch_floor=60.0, pitch_ceiling=300.0)
    formants = sound.to_formant_burg(
        time_step=hop_s,
        max_number_of_formants=5,
        maximum_formant=5300.0,
        window_length=0.0256,
        pre_emphasis_from=50.0,
    )

    numeric_names = [name for name in NASALITY_COLUMNS if name != "p0_id"]
    output = {name: np.full(datalen, np.nan) for name in numeric_names}
    output["p0_id"] = np.full(datalen, None, dtype=object)

    if intervals is None:
        intervals = [(0, duration_s * 1000.0, None)]
    measurement_frames = {}
    for start_ms, end_ms, _ in intervals:
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0
        if end_s <= start_s or 2 * scoot > end_s - start_s:
            continue
        if tpnum == 1:
            times = [(start_s + end_s) / 2.0]
        else:
            times = np.linspace(start_s, end_s, tpnum)
            times[0] += scoot
            times[-1] -= scoot
        interval_duration_ms = end_ms - start_ms
        for time_s in times:
            frame = int(round(time_s / hop_s))
            if 0 <= frame < datalen:
                measurement_frames[frame] = interval_duration_ms

    samples = sound.values[0]
    sample_rate = float(sound.sampling_frequency)
    window_samples = max(2, int(round(spectrum_window_s * sample_rate)))
    if window_samples % 2:
        window_samples += 1
    half_window = window_samples // 2
    hamming = np.hamming(window_samples)

    for index, interval_duration_ms in measurement_frames.items():
        time_s = index * hop_s
        output["vwl_duration"][index] = interval_duration_ms
        center_sample = int(round(time_s * sample_rate))
        start = center_sample - half_window
        stop = start + window_samples
        if start < 0 or stop > samples.size:
            continue

        f0 = pitch.get_value_at_time(time_s)
        f1 = formants.get_value_at_time(1, time_s)
        f2 = formants.get_value_at_time(2, time_s)
        f3 = formants.get_value_at_time(3, time_s)
        b1 = formants.get_bandwidth_at_time(1, time_s)
        b2 = formants.get_bandwidth_at_time(2, time_s)
        if not all(np.isfinite(x) for x in (f0, f1, f2, f3, b1, b2)) or f0 <= 0:
            continue

        frame = samples[start:stop] * hamming
        spectrum = parselmouth.Sound(frame, sample_rate).to_spectrum()
        ltas = praat.call(spectrum, "To Ltas (1-to-1)")

        _, a1 = _ltas_peak(ltas, f1 - 150.0, f1 + 150.0)
        _, a3 = _ltas_peak(ltas, f3 - 150.0, f3 + 150.0)
        h1_freq, h1_amp = _ltas_peak(ltas, 0.5 * f0, 1.5 * f0)
        h2_freq, h2_amp = _ltas_peak(ltas, 1.5 * f0, 2.5 * f0)
        _, h3_amp = _ltas_peak(ltas, 2.5 * f0, 3.5 * f0)
        p1_freq, p1_amp = _ltas_peak(ltas, p1search - 100.0, p1search + 100.0)
        if not all(
            np.isfinite(x)
            for x in (a1, a3, h1_freq, h1_amp, h2_freq, h2_amp, h3_amp,
                      p1_freq, p1_amp)
        ):
            continue

        # H2 can be P0 only when it is both stronger and below the script's
        # ``hip0`` guard.  H3 is deliberately not considered here.
        use_h2 = h2_amp >= h1_amp and h2_freq <= p0_ceiling_hz
        if use_h2:
            p0_id, p0_freq, p0_amp = "H2", h2_freq, h2_amp
            _, upper_amp = _ltas_peak(ltas, h2_freq + 0.5 * f0, h2_freq + 1.5 * f0)
            prominence = p0_amp - (h1_amp + upper_amp) / 2.0
        else:
            p0_id, p0_freq, p0_amp = "H1", h1_freq, h1_amp
            _, upper_amp = _ltas_peak(ltas, h1_freq + 0.5 * f0, h1_freq + 1.5 * f0)
            prominence = p0_amp - upper_amp

        a1p0_h1 = a1 - h1_amp
        a1p0_h2 = a1 - h2_amp
        a1p0_h3 = a1 - h3_amp
        a1p0 = a1 - p0_amp
        a1p1 = a1 - p1_amp
        p0_t1 = _chen_term(f1, b1, p0_freq)
        p0_t2 = _chen_term(f2, b2, p0_freq)
        p1_t1 = _chen_term(f1, b1, p1_freq)
        p1_t2 = _chen_term(f2, b2, p1_freq)

        values = {
            "amp_p0": p0_amp,
            "freq_p0": p0_freq,
            "p0_id": p0_id,
            "p0prominence": prominence,
            "a1p0_h1": a1p0_h1,
            "a1p0_h2": a1p0_h2,
            "a1p0_h3": a1p0_h3,
            "a1p0": a1p0,
            "a1p0_compensated": a1 - (p0_amp - p0_t1 - p0_t2),
            "freq_p1": p1_freq,
            "amp_p1": p1_amp,
            "a1p1": a1p1,
            "a1p1_Compensated": a1 - (p1_amp - p1_t1 - p1_t2),
            "a3p0": a3 - p0_amp,
        }
        for name, value in values.items():
            output[name][index] = value

    return pd.DataFrame({name: output[name] for name in NASALITY_COLUMNS})


def vspy_nasality(
    source,
    output_csv="nasality_output.csv",
    textgrid_dir=None,
    tier=None,
    p1search=950,
    tpnum=5,
    scoot=0.0,
):
    frames = []
    for path in resolve_paths(source):
        sound = parselmouth.Sound(str(path))
        datalen = int(round(sound.get_total_duration() * 1000)) + 1
        intervals = None
        if textgrid_dir is not None or tier is not None:
            if textgrid_dir is None or tier is None:
                raise ValueError("textgrid_dir and tier must be provided together")
            intervals = parse_textgrid(
                Path(textgrid_dir) / (path.stem + ".TextGrid"), tier
            )

        result = get_nasality(
            path,
            datalen=datalen,
            p1search=p1search,
            tpnum=tpnum,
            scoot=scoot,
            intervals=intervals,
        )
        selected = result["vwl_duration"].notna()
        result = result.loc[selected].copy()
        result.insert(0, "t_ms", result.index.to_numpy())
        result["filename"] = path.name
        if intervals is not None:
            labels = np.full(len(result), None, dtype=object)
            times = result["t_ms"].to_numpy()
            for start_ms, end_ms, label in intervals:
                labels[(times >= start_ms) & (times <= end_ms)] = label
            result[tier] = labels
        frames.append(result.reset_index(drop=True))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=("t_ms",) + NASALITY_COLUMNS + ("filename",))
    combined.to_csv(output_csv, index=False)
    return combined
