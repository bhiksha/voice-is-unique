"""Gender-classify the UNKNOWN-gender Common Voice speakers. RESUMABLE + crash-proof.

Vendored from the ~/data pipeline for self-contained corpus reproduction. IDENTICAL
logic; the only change is that the corpus root and wav root are taken from the
environment (CV_ROOT, CV_WAVROOT) so reproduce_corpus.py can point it at a rebuild
tree. Defaults reproduce the original ~/data locations.

Model: prithivMLmods/Common-Voice-Gender-Detection (Wav2Vec2ForSequenceClassification,
labels 0=female 1=male, 16 kHz). Each speaker has 10 aligned wavs in CV_WAVROOT/<sd>/.

Per speaker: classify all 10 utts (each clip forwarded individually -> softmax).
  (a) >=9/10 agree  -> tag that gender               (decision = tagged_<g>)
  (c) 5/5 split     -> tie                            (decision = tie)
  (b) 6,7,8 majority-> ambiguous; report majority, count, avg confidence over 10

Canonical incremental store / resume point:  <CV_ROOT>/unknown/estimated_gender
  - one TSV line APPENDED per speaker the instant it is classified (atomic O_APPEND
    write, lock-free, < PIPE_BUF). This is the human-readable record AND the resume
    source: on (re)start, speakers already present are skipped. A hang/OOM/kill thus
    costs at most the in-flight speaker.
Full per-utt detail (softmax probs per clip) is also kept in _gender_work/<sd>.json.
'avg_conf' = signed agreement score (see signed_avg_conf).

NOTE: run with --workers 4. The 8-worker run has a documented multiprocessing.Pool
OOM-stall on the last few speakers; 4 workers avoids it. If it ever hangs on the
tail, kill and re-run -- it resumes from estimated_gender.
"""
import os, csv, glob, json, argparse, collections
import multiprocessing as mp
import numpy as np
import soundfile as sf

ROOT = os.environ.get("CV_ROOT", "/home/bhiksha/data/commonvoice")
CORP = os.environ.get("CV_WAVROOT", "/home/bhiksha/cv_align/corpus")
WORKDIR = f"{ROOT}/_gender_work"
EST = f"{ROOT}/unknown/estimated_gender"          # canonical incremental store + resume
EST_COLS = ["speaker_dir", "client_id", "n_female", "n_male",
            "majority", "majority_count", "avg_conf", "decision"]
MODEL = "prithivMLmods/Common-Voice-Gender-Detection"
SR = 16000


def signed_avg_conf(utts, nf, nm):
    """Mean over the 10 clips of (+conf if clip votes the MAJORITY gender, else -conf).
    Confident male & female clips cancel -> ~0 for split/tie speakers, ~0.99 for clean
    ones. Tie (nf==nm) uses female as the reference, so it still cancels to ~0."""
    maj = "female" if nf >= nm else "male"
    s = 0.0
    for u in utts:
        conf = max(u["p_female"], u["p_male"])
        s += conf if u["pred"] == maj else -conf
    return s / len(utts)

_fe = None
_model = None
_estfd = None


def _est_line(r):
    return ("\t".join([r["speaker_dir"], r["client_id"], str(r["n_female"]), str(r["n_male"]),
                       r["majority"], str(r["majority_count"]),
                       f"{r['avg_conf']:.4f}", r["decision"]]) + "\n")


def _init(threads):
    global _fe, _model, _estfd
    import torch
    torch.set_num_threads(threads)
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    _fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    _model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL).eval()
    _estfd = os.open(EST, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)  # shared append


def _classify_speaker(task):
    sd, client_id, wavs = task
    import torch
    recs = []
    nf = nm = 0
    for w in sorted(wavs):
        a, sr = sf.read(w)
        if a.ndim > 1:
            a = a.mean(1)
        inp = _fe(a, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = _model(**inp).logits
        prob = torch.softmax(logits, dim=-1)[0].numpy()
        pf, pm = float(prob[0]), float(prob[1])
        pred = "male" if pm >= pf else "female"
        nm += pred == "male"
        nf += pred == "female"
        recs.append({"clip": os.path.basename(w), "pred": pred,
                     "p_female": round(pf, 4), "p_male": round(pm, 4)})
    maj = "male" if nm > nf else ("female" if nf > nm else "tie")
    mc = max(nm, nf)
    rec = {"speaker_dir": sd, "client_id": client_id, "n_female": nf, "n_male": nm,
           "majority": maj, "majority_count": mc, "avg_conf": signed_avg_conf(recs, nf, nm),
           "decision": (f"tagged_{maj}" if mc >= 9 else "tie" if mc == 5 else "ambiguous"),
           "utts": recs}
    tmp = f"{WORKDIR}/{sd}.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, f"{WORKDIR}/{sd}.json")          # atomic detail persist
    os.write(_estfd, _est_line(rec).encode())        # atomic append to canonical record
    return sd


def _done_from_est():
    """speaker_dirs already recorded in estimated_gender (skip header)."""
    if not os.path.exists(EST):
        return set()
    done = set()
    with open(EST) as fh:
        for i, ln in enumerate(fh):
            if i == 0 and ln.startswith("speaker_dir\t"):
                continue
            if ln.strip():
                done.add(ln.split("\t", 1)[0])
    return done


def _ensure_est_header_and_seed():
    """Create estimated_gender with header if absent; migrate any pre-existing
    _gender_work JSONs into it so completed work survives the format switch."""
    if not os.path.exists(EST) or os.path.getsize(EST) == 0:
        os.makedirs(os.path.dirname(EST), exist_ok=True)
        with open(EST, "w") as fh:
            fh.write("\t".join(EST_COLS) + "\n")
    done = _done_from_est()
    seeded = 0
    with open(EST, "a") as fh:
        for p in sorted(glob.glob(f"{WORKDIR}/*.json")):
            sd = os.path.basename(p)[:-5]
            if sd in done:
                continue
            try:
                r = json.load(open(p))
            except Exception:
                continue
            fh.write(_est_line(r)); done.add(sd); seeded += 1
    if seeded:
        print(f"seeded {seeded} pre-existing results into estimated_gender", flush=True)
    return done


def aggregate():
    results = []
    for p in glob.glob(f"{WORKDIR}/*.json"):
        try:
            results.append(json.load(open(p)))
        except Exception:
            pass
    results.sort(key=lambda r: r["speaker_dir"])
    with open(f"{ROOT}/gender_predictions.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(EST_COLS)
        for r in results:
            w.writerow([r["speaker_dir"], r["client_id"], r["n_female"], r["n_male"],
                        r["majority"], r["majority_count"], f"{r['avg_conf']:.4f}", r["decision"]])
    json.dump(results, open(f"{ROOT}/gender_predictions_detail.json", "w"))

    def dump(name, rs):
        with open(f"{ROOT}/{name}", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["speaker_dir", "majority", "n_majority/10", "avg_conf",
                        "n_female", "n_male", "client_id"])
            for r in rs:
                w.writerow([r["speaker_dir"], r["majority"], r["majority_count"],
                            f"{r['avg_conf']:.4f}", r["n_female"], r["n_male"], r["client_id"]])
    amb = sorted((r for r in results if r["decision"] == "ambiguous"),
                 key=lambda r: (-r["majority_count"], -r["avg_conf"]))
    ties = sorted((r for r in results if r["decision"] == "tie"), key=lambda r: -r["avg_conf"])
    dump("gender_ambiguous.tsv", amb)
    dump("gender_ties.tsv", ties)
    dec = collections.Counter(r["decision"] for r in results)
    print(f"\n=== AGGREGATED {len(results)} speakers ===", flush=True)
    for k in sorted(dec):
        print(f"  {k}: {dec[k]}")
    print(f"ambiguous={len(amb)} ties={len(ties)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORKDIR, exist_ok=True)

    if a.aggregate_only:
        aggregate()
        return

    from huggingface_hub import hf_hub_download
    for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
        hf_hub_download(MODEL, f)

    done = _ensure_est_header_and_seed()
    rows = list(csv.DictReader(open(f"{ROOT}/speaker_map.tsv"), delimiter="\t"))
    unk = [r for r in rows if r["gender_dir"] == "unknown"]
    if a.limit:
        unk = unk[:a.limit]
    todo = [r for r in unk if r["speaker_dir"] not in done]
    tasks = [(r["speaker_dir"], r["client_id"],
              glob.glob(f"{CORP}/{r['speaker_dir']}/*.wav")) for r in todo]
    print(f"unknown={len(unk)} done={len(done)} todo={len(tasks)} | "
          f"workers={a.workers} threads={a.threads}", flush=True)

    if tasks:
        with mp.Pool(a.workers, initializer=_init, initargs=(a.threads,),
                     maxtasksperchild=200) as pool:
            for i, sd in enumerate(pool.imap_unordered(_classify_speaker, tasks, chunksize=2), 1):
                if i % 200 == 0:
                    print(f"  {i}/{len(tasks)} (last {sd})", flush=True)

    final_done = _done_from_est()
    missing = [r["speaker_dir"] for r in unk if r["speaker_dir"] not in final_done]
    print(f"recorded={len(final_done)} missing={len(missing)}", flush=True)
    if missing:
        print("  still-missing (re-run to finish):", missing[:20], flush=True)
    aggregate()


if __name__ == "__main__":
    main()
