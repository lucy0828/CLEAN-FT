"""Participant-clustered uncertainty for the OOD detection AUROC
-> results/ood_clustered.csv.

How to run (CPU; reads results/raw/stage3/ood_* and results/stage3/annotations.csv):
    uv run python -m src.stage3.ood_uncertainty

The 844 screened phases come from 21 participants and the 34 annotated ones
from 7, so posts are not independent draws. For every method with a phase
table and every view (annotator1 / annotator2 / pooled) this reports, beside
the per-phase bootstrap `results/ood_transfer.csv` already carries:

* a PARTICIPANT-clustered bootstrap of the AUROC (2,000 draws of participants
  with replacement, every phase of a drawn participant kept), on the screened
  corpus (screen-negatives as 0) and on the annotated phases alone;
* leave-one-participant-out: the AUROC with each participant of the annotated
  set held out in turn (min / mean / max over the seven), which is what a
  single participant contributing 16 of the 34 phases does to the number;
* the per-annotator class counts, which the OOD table carries only in
  aggregate.
"""

from __future__ import annotations

import argparse
import glob as globlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.stage3.ood_detection import VIEWS, load_annotations, supervision
from src.utils.common import RESULTS, RAW_STAGE3, resolve_model

OUT_CSV = RESULTS / "ood_clustered.csv"
N_BOOT = 2000
SEED = 0


def phase_tables(model: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(globlib.glob(str(RAW_STAGE3 / f"ood_*_{model_label(model)}.parquet"))):
        frame = pd.read_parquet(path)
        for method, g in frame.groupby("method"):
            out[method] = g
    return out


def model_label(model: str) -> str:
    from src.utils.common import model_label as _ml

    return _ml(model)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def clustered(frame: pd.DataFrame, rng) -> tuple[float, float, float]:
    """AUROC and its participant-bootstrap interval on `frame` (y, score, pid)."""
    groups = {p: g for p, g in frame.groupby("pid")}
    pids = list(groups)
    point = auroc(frame["y"], frame["score"])
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(pids, size=len(pids), replace=True)
        sample = pd.concat([groups[p] for p in pick], ignore_index=True)
        v = auroc(sample["y"], sample["score"])
        if np.isfinite(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return point, float(lo), float(hi)


def phase_evidence(method: str, model: str, annotations, view: str, phases_text):
    """Per-phase word AUPRC and ΔAOPC margin for one method under one view,
    with the participant of each phase -- the per-phase inputs
    src.stage3.ood_detection.collect_metrics averages."""
    import json

    from src.stage2.word_deletion import _margins
    from src.stage3.ood_detection import (delete_path, phase_word_auprc,
                                          word_path)

    sup = supervision(annotations, view)
    positives = sup[sup["label"] == 1]
    w_path, d_path = word_path(method, model), delete_path(method, model)
    words_by_phase = {pid: g.sort_values("word_index")["score"].to_numpy(dtype=float)
                      for pid, g in pd.read_parquet(w_path).groupby("phase_id")} \
        if w_path.exists() else {}
    auprc = {}
    for r in positives.itertuples():
        spans = json.loads(r.span_chars)
        if not spans or r.phase_id not in words_by_phase:
            continue
        ap, _, _ = phase_word_auprc(phases_text[r.phase_id], words_by_phase[r.phase_id], spans)
        if np.isfinite(ap):
            auprc[r.phase_id] = ap
    margin = {}
    if d_path.exists() and len(positives):
        d = pd.read_parquet(d_path)
        d = d[d["id"].isin(set(positives["phase_id"]))]
        d = d[(d["intervention"] == "word_delete") & (~d["small_logit_gap"].astype(bool))
              & (d["clean_empty_logit_gap"] > 0)]
        tgt = d[d["control"] == "target"].groupby("id")["normalized_aopc"].first()
        ctl = d[d["control"] == "random_words"].groupby("id")["normalized_aopc"].mean()
        for pid in tgt.index.intersection(ctl.index):
            margin[pid] = float(tgt[pid] - ctl[pid])
    return auprc, margin


def lopo_mean(values: dict, pid_of: dict) -> tuple[float, float, float, int]:
    """(min, mean, max, n participants) of the mean over phases with each
    participant held out in turn."""
    pids = sorted({pid_of[k] for k in values if k in pid_of})
    outs = []
    for p in pids:
        rest = [v for k, v in values.items() if pid_of.get(k) != p]
        if rest:
            outs.append(float(np.mean(rest)))
    if not outs:
        return float("nan"), float("nan"), float("nan"), 0
    return float(min(outs)), float(np.mean(outs)), float(max(outs)), len(outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", default=["gemma2", "llama"])
    args = ap.parse_args()
    annotations = load_annotations(required=True)
    from src.stage3.ood_detection import load_phases
    from src.stage3.prepare_conversations import CORPORA

    phases, _turns = load_phases(list(CORPORA))
    phases_text = phases.set_index("phase_id")["text"].to_dict()
    rng = np.random.default_rng(SEED)
    rows = []
    for model in [resolve_model(m) for m in args.model]:
        tables = phase_tables(model)
        for method, frame in tables.items():
            frame = frame.rename(columns={"nickname": "pid"})
            for view in VIEWS:
                sup = supervision(annotations, view)
                lab = dict(zip(sup["phase_id"], sup["label"]))
                # pooled view lists a disputed phase twice; keep one row per
                # phase with its (possibly duplicated) supervision rows
                ann = frame[frame["phase_id"].isin(lab)].copy()
                ann = ann.merge(sup[["phase_id", "label"]], on="phase_id")
                ann["y"] = ann["label"].astype(int)
                neg = frame[~frame["phase_id"].isin(set(annotations["phase_id"]))].copy()
                neg["y"] = 0
                screened = pd.concat([ann, neg], ignore_index=True)

                a_pt, a_lo, a_hi = clustered(screened[["y", "score", "pid"]], rng)
                b_pt, b_lo, b_hi = clustered(ann[["y", "score", "pid"]], rng)
                lopo = []
                for pid in ann["pid"].unique():
                    rest = ann[ann["pid"] != pid]
                    lopo.append(auroc(rest["y"], rest["score"]))
                lopo = np.asarray([v for v in lopo if np.isfinite(v)])
                pid_of = dict(zip(frame["phase_id"], frame["pid"]))
                ev_auprc, ev_margin = phase_evidence(method, model, annotations, view,
                                                     phases_text)
                a_min, a_mean, a_max, a_n = lopo_mean(ev_auprc, pid_of)
                m_min, m_mean, m_max, m_n = lopo_mean(ev_margin, pid_of)
                rows.append({
                    "auprc_lopo_min": a_min, "auprc_lopo_mean": a_mean,
                    "auprc_lopo_max": a_max, "auprc_n_phases": len(ev_auprc),
                    "aopc_lopo_min": m_min, "aopc_lopo_mean": m_mean,
                    "aopc_lopo_max": m_max, "aopc_n_phases": len(ev_margin),
                    "method": method, "model": model, "annotator": view,
                    "n_participants_screened": int(screened["pid"].nunique()),
                    "n_participants_annotated": int(ann["pid"].nunique()),
                    "n_pos": int(ann["y"].sum()),
                    "n_neg_annotated": int((ann["y"] == 0).sum()),
                    "n_neg_screen": int(len(neg)),
                    "auroc_screened": a_pt, "auroc_screened_lo_cluster": a_lo,
                    "auroc_screened_hi_cluster": a_hi,
                    "auroc_annotated": b_pt, "auroc_annotated_lo_cluster": b_lo,
                    "auroc_annotated_hi_cluster": b_hi,
                    "lopo_min": float(lopo.min()) if len(lopo) else np.nan,
                    "lopo_mean": float(lopo.mean()) if len(lopo) else np.nan,
                    "lopo_max": float(lopo.max()) if len(lopo) else np.nan,
                    "lopo_n": int(len(lopo)),
                })
                r = rows[-1]
                print(f"  {model} {method:<22} {view:<10} screened "
                      f"{a_pt:.3f} [{a_lo:.3f}, {a_hi:.3f}]  annotated "
                      f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]  LOPO "
                      f"{r['lopo_min']:.3f}–{r['lopo_max']:.3f}  "
                      f"(+{r['n_pos']}/-{r['n_neg_annotated']} annotated)  "
                      f"AUPRC LOPO {a_min:.3f}–{a_max:.3f}  ΔAOPC LOPO {m_min:.2f}–{m_max:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.name} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
