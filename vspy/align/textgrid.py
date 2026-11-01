import numpy as np




def parse_textgrid(tg_path, tier_name, round_times=True):
    raw = open(tg_path, "rb").read()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    lines = [l.strip() for l in raw.decode(enc).splitlines()]

    intervals = []
    in_tier = False
    current = {}

    for line in lines:
        if f'name = "{tier_name}"' in line:
            in_tier = True
        if not in_tier:
            continue
        if line.startswith("xmin") and "=" in line:
            current["xmin"] = float(line.split("=")[1])
        elif line.startswith("xmax") and "=" in line:
            current["xmax"] = float(line.split("=")[1])
        elif line.startswith("text") and "=" in line:
            label = line.split("=")[1].strip().strip('"')
            current["text"] = label
            if label != "":
                intervals.append((
                    round(current["xmin"] * 1000) if round_times else current["xmin"] * 1000,
                    round(current["xmax"] * 1000) if round_times else current["xmax"] * 1000,
                    label,
                ))
            current = {}
        elif in_tier and line.startswith("item ["):
            break

    return intervals




def label_frames(intervals, datalen, frameshift_ms=1):
    """Label frames at k * frameshift_ms within [start_ms, end_ms)."""
    if not np.isfinite(frameshift_ms) or frameshift_ms <= 0:
        raise ValueError("frameshift_ms must be finite and positive")
    times_ms = np.arange(datalen) * frameshift_ms
    labels = np.full(datalen, None, dtype=object)
    for start_ms, end_ms, label in intervals:
        start, end = np.searchsorted(times_ms, [start_ms, end_ms], side="left")
        labels[start:end] = label
    return labels
