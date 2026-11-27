"""Extract aligned VSpy frames for a corpus or a single recording. Run python -m evaluate.extract --help from the repository root."""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow.parquet as pq

from vspy import vspy


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
    ("B4_praat",  "pB4",    "B4_praat"),
    ("B1_snack",  "sB1",    "B1_snack"),
    ("B2_snack",  "sB2",    "B2_snack"),
    ("B3_snack",  "sB3",    "B3_snack"),
    ("B4_snack",  "sB4",    "B4_snack"),
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
    # voice-quality / periodicity measures
    ("CPP",       "CPP",    "CPP"),
    ("Energy",    "Energy", "Energy"),
    ("HNR05",     "HNR05",  "HNR05"),
    ("HNR15",     "HNR15",  "HNR15"),
    ("HNR25",     "HNR25",  "HNR25"),
    ("HNR35",     "HNR35",  "HNR35"),
    ("SoE",       "soe",    "SoE"),
    ("epoch",     "epoch",  "epoch"),
    ("shr",       "SHR",    "SHR"),
]

F0_PAIRS = [
    ("f0_straight", "strF0", "f0_straight"),
    ("f0", "shrF0", "f0_shr"),
    ("f0_snack", "sF0", "f0_snack"),
    ("f0_praat", "pF0", "f0_praat"),
]

def select_features(reference_columns):
    """Select registry extractors and output columns from the VoiceSauce header."""
    selected = [v for v, ref, _ in FEATURE_PAIRS + F0_PAIRS if ref in reference_columns]
    if not selected:
        raise ValueError("No comparable measurements in the VoiceSauce reference")
    modules = {
        "f0_straight": {"f0_straight"}, "f0_snack": {"f0_snack"}, "f0_praat": {"f0_praat"},
        "shrp": {"shr", "f0"},
        "formants_praat": {f"{kind}{i}_praat" for kind in "FB" for i in range(1, 5)},
        "formants_snack": {f"{kind}{i}_snack" for kind in "FB" for i in range(1, 5)},
    }
    requested = {name for name, outputs in modules.items() if outputs.intersection(selected)}
    direct = set().union(*modules.values())
    if set(selected) - direct:
        # The existing API computes derived measures using STRAIGHT pitch and Praat formants.
        requested.update(("f0_straight", "formants_praat"))
    return sorted(requested), selected


def read_references(path: Path, filenames) -> dict:
    """Read just the reference frame times, boundaries and labels for the input WAVs."""
    required = {"Filename", "t_ms", "seg_Start", "seg_End"}
    columns = required | {"Label"}
    options = dict(sep="\t", keep_default_na=False)
    header = pd.read_csv(path, nrows=0, **options)
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")

    groups = defaultdict(list)
    with pd.read_csv(path, chunksize=100_000, usecols=lambda c: c in columns,
                     **options) as chunks:
        for chunk in chunks:
            selected = chunk.loc[chunk["Filename"].isin(filenames)]
            for name, rows in selected.groupby("Filename", sort=False):
                groups[name].append(rows)
    missing = set(filenames).difference(groups)
    if missing:
        raise ValueError(f"No VoiceSauce reference found for: {sorted(missing)}")
    references = {}
    for name, parts in groups.items():
        table = pd.concat(parts, ignore_index=True)
        for column in ("seg_Start", "seg_End", "t_ms"):
            table[column] = pd.to_numeric(table[column], errors="raise")
        references[name] = table
    return references, select_features(header.columns)


def find_wavs(corpus: Path, recording_id=None) -> list:
    """Match complete recording IDs, including any suffix before the extension."""
    wavs = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"
                  and (recording_id is None or p.stem == recording_id))
    if not wavs or len({p.stem for p in wavs}) != len(wavs):
        raise ValueError(f"Expected WAV files with unique recording IDs in {corpus}")
    return wavs


def align_frames(reference: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Match frames by millisecond timestamp, keeping each reference label at shared times."""
    aligned = reference.merge(features, on="t_ms", how="inner", validate="many_to_one")
    if aligned.empty:
        raise ValueError("The reference table and VSpy output have no timestamps in common")
    return aligned.sort_values("t_ms", kind="stable").reset_index(drop=True)


def extract(wav: Path, reference: pd.DataFrame, output: Path, modules, columns) -> None:
    recording_id = wav.stem
    if output.suffix.lower() != ".parquet":
        raise ValueError("Output filename must end in .parquet")
    if output.exists():
        raise FileExistsError(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    # VSpy requires an output path. Keep its raw features in a temporary directory.
    with TemporaryDirectory(dir=output.parent) as directory:
        tmp = Path(directory)
        features = vspy(source=wav, output_file=tmp / "raw.parquet", features=modules,
                        frameshift_ms=1, textgrid_dir=None)
        features = features[["filename", "t_ms", *columns]]
        aligned = align_frames(reference, features)
        aligned.insert(0, "recording_id", recording_id)
        staged = tmp / "aligned.parquet"
        aligned.to_parquet(staged, index=False)
        # Link the finished file into place. This fails if the destination already exists.
        os.link(staged, output)

    print(f"{recording_id}: {len(reference)} reference rows, {len(aligned)} retained", flush=True)


def run(corpus, recording_id, output, jobs, resume):
    if jobs < 1 or (output is not None and recording_id is None):
        raise ValueError("Use a positive job count; an explicit output requires a recording ID")
    wavs = find_wavs(corpus, recording_id)
    refs, (modules, columns) = read_references(corpus / "output.txt", {w.stem + ".mat" for w in wavs})
    print(f"Extractors: {modules}. Saving {len(columns)} measurements.", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = {}
        for wav in wavs:
            target = output or corpus / (wav.stem + ".parquet")
            if resume and target.is_file():
                # Each recording is published atomically; resume only an unchanged corpus.
                expected = {"recording_id", "filename", *refs[wav.stem + ".mat"].columns, *columns}
                if set(pq.read_schema(target).names) != expected:
                    raise ValueError(f"Existing output has a different feature set: {target}")
                saved = pd.read_parquet(target, columns=["recording_id", "filename", "t_ms"])
                if saved.empty or not (saved.recording_id.eq(wav.stem) & saved.filename.eq(wav.name)).all():
                    raise ValueError(f"Unexpected recording in {target}")
                print(f"Reusing {target.name}", flush=True)
                continue
            if target.exists():
                raise FileExistsError(target)
            tasks[pool.submit(extract, wav, refs[wav.stem + ".mat"], target, modules, columns)] = wav
        for task in as_completed(tasks):
            try:
                task.result()
            except Exception as exc:
                raise RuntimeError(f"Extraction failed for {tasks[task]}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory containing output.txt and WAV files")
    parser.add_argument("recording_id", nargs="?", help="Optional single WAV stem (default: all WAVs)")
    parser.add_argument("output", type=Path, nargs="?", help="Optional single-recording output file")
    parser.add_argument("--jobs", type=int, default=1, help="Recordings to process concurrently")
    parser.add_argument("--resume", action="store_true", help="Reuse completed files from the same run and inputs")
    args = parser.parse_args()
    run(args.corpus, args.recording_id, args.output, args.jobs, args.resume)
