#!/usr/bin/env python
"""Download 10,000 Common Voice 17.0 English speakers x 10 clips (each >= 4 s) into a
gender-branched tree under ~/data/commonvoice:

    commonvoice/
      female/   f_<grp>/  f_<grp>_<idx>/   <10 .mp3>     (self-reported female)
      male/     m_<grp>/  m_<grp>_<idx>/   <10 .mp3>     (self-reported male)
      unknown/  u_<grp>/  u_<grp>_<idx>/   <10 .mp3>     (no declared gender; fills to 10k)
      speaker_map.tsv      (speaker_dir -> client_id -> its 10 clip filenames)

Only the TRAIN split is used, because that is the portion whose audio is packaged in
the public HF mirror's tars (validated-only clips have no fetchable audio). All
self-reported-gender speakers that qualify go into female/ + male/; the rest of the
10,000 is filled from the unlabeled ("unknown") pool. 100 speakers per group dir.

Auth: uses a cached HF login (~/.cache/huggingface/token) or the HF_TOKEN env var.
Deterministic: speakers sorted by client_id; first 10 clips per speaker by sorted id.
"""
import os, shutil, tarfile, collections, csv, math
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, login

ROOT   = os.path.expanduser("~/data/commonvoice")
RID    = "fsicoli/common_voice_17_0"
LANG   = "en"
N_TOTAL, N_CLIPS, MIN_MS = 10000, 10, 4000
MALE   = {"male", "male_masculine"}
FEMALE = {"female", "female_feminine"}
BRANCHES = [("female", "f"), ("male", "m"), ("unknown", "u")]


def qualifying_speakers():
    """Speakers in train with >= N_CLIPS clips of >= MIN_MS, with their sorted clips + sex."""
    tr = pd.read_csv(hf_hub_download(RID, f"transcript/{LANG}/train.tsv", repo_type="dataset"),
                     sep="\t", usecols=["client_id", "path", "gender"],
                     quoting=3, dtype=str, on_bad_lines="skip")
    dur = pd.read_csv(hf_hub_download(RID, f"transcript/{LANG}/clip_durations.tsv", repo_type="dataset"), sep="\t")
    dmap = dict(zip(dur.iloc[:, 0], dur.iloc[:, 1]))
    tr["dur"] = tr["path"].map(dmap)
    tr = tr[tr["dur"] >= MIN_MS]
    cps = tr.groupby("client_id").size()
    qual = set(cps[cps >= N_CLIPS].index)
    tr = tr[tr.client_id.isin(qual)]
    clips = collections.defaultdict(list)
    for r in tr.itertuples():
        clips[r.client_id].append(r.path)
    g1 = tr.groupby("client_id")["gender"].first().str.lower()
    sex = {c: ("M" if g1.get(c) in MALE else "F" if g1.get(c) in FEMALE else "U") for c in qual}
    return clips, sex


def build_plan(clips, sex):
    """Select 10k speakers (all gendered + unknown fill); build tree + clip->target + map."""
    males   = sorted(c for c in sex if sex[c] == "M")
    females = sorted(c for c in sex if sex[c] == "F")
    unknown = sorted(c for c in sex if sex[c] == "U")
    n_unknown = max(0, N_TOTAL - len(males) - len(females))
    selected = {"female": females, "male": males, "unknown": unknown[:n_unknown]}
    print(f"qualifying: M={len(males)} F={len(females)} U={len(unknown)}")
    print(f"selected:   female={len(females)} male={len(males)} unknown={len(selected['unknown'])} "
          f"-> total {sum(len(v) for v in selected.values())}")

    for top, _ in BRANCHES:
        shutil.rmtree(os.path.join(ROOT, top), ignore_errors=True)
    clip2tgt, maprows = {}, []
    for top, pre in BRANCHES:
        spk = selected[top]
        print(f"  {top}: {len(spk)} speakers -> {pre}_1..{pre}_{math.ceil(len(spk)/100) or 0}")
        for k, s in enumerate(spk):
            gi, li = k // 100 + 1, k % 100 + 1
            name = f"{pre}_{gi}_{li}"
            sdir = os.path.join(ROOT, top, f"{pre}_{gi}", name)
            os.makedirs(sdir, exist_ok=True)
            cl = sorted(clips[s])[:N_CLIPS]
            for clip in cl:
                clip2tgt[clip] = os.path.join(sdir, clip)
            maprows.append((name, f"{top}/{pre}_{gi}/{name}", top, s, len(cl), ",".join(cl)))
    with open(os.path.join(ROOT, "speaker_map.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_dir", "rel_path", "gender_dir", "client_id", "n_clips", "clips"])
        w.writerows(maprows)
    print(f"speakers: {len(maprows)} | clips to fetch: {len(clip2tgt)} | map -> {ROOT}/speaker_map.tsv")
    return clip2tgt


def download(clip2tgt):
    """Fetch CV17 en train tars one at a time, extract the selected clips, delete each tar."""
    TMP = os.path.expanduser("~/data/_cv17tmp")
    os.makedirs(TMP, exist_ok=True)
    want = set(clip2tgt)
    api = HfApi()
    tars = sorted(s.rfilename for s in api.repo_info(RID, repo_type="dataset").siblings
                  if s.rfilename.startswith(f"audio/{LANG}/train/"))
    found = 0
    for ti, t in enumerate(tars, 1):
        if found >= len(want):
            break
        local = hf_hub_download(RID, t, repo_type="dataset", local_dir=TMP)
        with tarfile.open(local) as tf:
            for mem in tf:
                name = os.path.basename(mem.name)
                tgt = clip2tgt.get(name)
                if tgt and not (os.path.exists(tgt) and os.path.getsize(tgt) > 0):
                    f = tf.extractfile(mem)
                    if f:
                        with open(tgt, "wb") as o:
                            o.write(f.read())
                        found += 1
        os.remove(local)
        print(f"[{ti}/{len(tars)}] {os.path.basename(t)}: {found}/{len(want)} clips", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    missing = [c for c in want if not (os.path.exists(clip2tgt[c]) and os.path.getsize(clip2tgt[c]) > 0)]
    print(f"DONE: {len(want) - len(missing)}/{len(want)} clips downloaded; missing={len(missing)}")
    if missing:
        open("/tmp/cv17_10k_missing.txt", "w").write("\n".join(sorted(missing)))


def main():
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    clips, sex = qualifying_speakers()
    clip2tgt = build_plan(clips, sex)
    download(clip2tgt)


if __name__ == "__main__":
    main()
