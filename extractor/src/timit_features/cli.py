"""Command-line entry point for the TIMIT 40-feature extractor.

  timit-features <timit_root> [--out DIR] [--limit N] [--speaker ID]
                 [--utt PATH] [--jobs N]

Determinism: utterances are processed independently and the consolidated table is
sorted by rel_path, so results never depend on --jobs or worker order.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from timit_features.config import CONFIG, DEFAULT_OUTPUT_ROOT
from timit_features.extract import safe_extract_utterance
from timit_features import writers


def discover_utterances(root: Path, speaker: str | None = None) -> list[Path]:
    wavs = sorted(root.rglob("*.WAV"))
    if speaker:
        wavs = [w for w in wavs if w.parent.name == speaker]
    return wavs


def _out_json(wav: Path, root: Path, out: Path) -> Path:
    return out / Path(wav).resolve().relative_to(Path(root).resolve()).with_suffix(".json")


def _process_and_write(wav: Path, root: Path, out: Path):
    """Worker: extract one utterance (never raises) and write its JSON immediately."""
    r = safe_extract_utterance(wav, root)
    writers.write_utterance_json(r, out)
    return r.rel_path, r.decode_ok, r.error


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="TIMIT 40-feature utterance extractor")
    ap.add_argument("timit_root", type=Path, help="root of the TIMIT corpus (has TRAIN/ TEST/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT, help="output dir")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N utterances")
    ap.add_argument("--speaker", default=None, help="only this speaker id (e.g. FCJF0)")
    ap.add_argument("--utt", type=Path, default=None, help="process exactly one utterance (.WAV)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute utterances whose JSON already exists (default: skip)")
    args = ap.parse_args(argv)

    root = args.timit_root
    args.out.mkdir(parents=True, exist_ok=True)
    if args.utt:
        wavs = [args.utt]
    else:
        wavs = discover_utterances(root, args.speaker)
        if args.limit > 0:
            wavs = wavs[: args.limit]
    if not wavs:
        print("No utterances found.", file=sys.stderr)
        sys.exit(1)

    total = len(wavs)
    if not args.overwrite:                       # resumable: skip already-written
        wavs = [w for w in wavs if not _out_json(w, root, args.out).exists()]
    print(f"{total} utterance(s); {total - len(wavs)} already done; "
          f"processing {len(wavs)} → {args.out} (jobs={args.jobs})", flush=True)

    # Write CONFIG up front so the parameter set is recorded even if the run is cut short.
    writers.write_config(CONFIG, args.out)

    fn = partial(_process_and_write, root=root, out=args.out)
    done = fails = 0
    if args.jobs > 1 and len(wavs) > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for rel, ok, err in pool.imap_unordered(fn, wavs, chunksize=1):
                done += 1
                fails += (not ok)
                if not ok:
                    print(f"  ! {rel}: {err}", flush=True)
                if done % 50 == 0 or done == len(wavs):
                    print(f"  [{done}/{len(wavs)}] (decode-fails so far: {fails})", flush=True)
    else:
        for i, w in enumerate(wavs, 1):
            rel, ok, err = fn(w)
            fails += (not ok)
            if not ok:
                print(f"  ! {rel}: {err}", flush=True)
            if i % 50 == 0 or i == len(wavs):
                print(f"  [{i}/{len(wavs)}]", flush=True)

    # Build the consolidated table + manifest from ALL JSONs present (this run + prior).
    records = writers.read_records(args.out)
    writers.write_table(writers.build_dataframe(records), args.out)
    writers.write_manifest(records, CONFIG, args.out)
    decoded = sum(1 for r in records if r.decode_ok)
    print(f"Done. Total utterances on disk: {len(records)} ({decoded} decoded). "
          f"Wrote all_utterances.parquet/.csv + MANIFEST.json to {args.out}", flush=True)


if __name__ == "__main__":
    main()
