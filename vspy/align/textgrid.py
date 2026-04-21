import numpy as np 

# define textgrid parsing function 

# reads file, finds tier by name, and return a list of (start_ms, end_ms, and label)
def parse_textgrid(tg_path, tier_name):
    with open(tg_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

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
                    round(current["xmin"] * 1000),
                    round(current["xmax"] * 1000),
                    label,
                ))
            current = {}
        elif in_tier and line.startswith("item ["):
            break  # hit the next tier, stop

    return intervals


# labeling function that returns a string of length datalen
# maps labels onto each frame, returns 'none' if there is no label
def label_frames(intervals, datalen):
    labels = np.full(datalen, None, dtype=object)
    for start_ms, end_ms, label in intervals:
        for i in range(start_ms, min(end_ms, datalen)):
            labels[i] = label
    return labels