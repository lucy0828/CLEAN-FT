"""Participant-clustered intervals for the OOD evidence metrics, and the
transferred-threshold sensitivity against each annotator
-> results/ood_clustered_evidence.csv.

How to run (CPU; reads results/raw/stage3/* and results/stage3/annotations.csv):
    uv run python -m src.stage3.ood_evidence_ci

WHY (2026-10-06, reviewer response)
-----------------------------------
The paper's Table 2 gives OOD word AUPRC and ΔAOPC on the 34 annotated
phases with no uncertainty, and the phases come from 7 participants (one
contributes 16 of 34), so a per-phase bootstrap understates it. For every
method, backbone and annotator view this reports:

* word AUPRC and ΔAOPC with a PARTICIPANT-clustered bootstrap (2,000 draws of
  participants with replacement; every annotated phase of a drawn participant
  kept), beside the per-phase bootstrap src.stage3.ood_detection publishes;
* sensitivity and specificity on the 34 phases at the threshold transferred
  from LonXplain (no re-cut), against each annotator's labels;
* how many of the 810 screen-negative phases the detector flags at that
  threshold -- the detector-side bound on what the LLM screen may have missed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.stage3.ood_detection import VIEWS, load_annotations, supervision
from src.stage3.ood_uncertainty import phase_evidence, phase_tables
from src.utils.common import RESULTS, resolve_model

OUT_CSV = RESULTS / "ood_clustered_evidence.csv"
N_BOOT = 2000
SEED = 0


def clustered_mean(values: dict, pid_of: dict, rng) -> tuple[float, float, float, int, int]:
    """(mean, lo, hi, n phases, n participants): mean over phases, resampling
    PARTICIPANTS with replacement."""
    by_pid: dict = {}
    for k, v in values.items():
        if k in pid_of and np.isfinite(v):
            by_pid.setdefault(pid_of[k], []).append(float(v))
    pids = sorted(by_pid)
    if not pids:
        return float("nan"), float("nan"), float("nan"), 0, 0
    flat = np.concatenate([by_pid[p] for p in pids])
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(pids, size=len(pids), replace=True)
        draws.append(np.mean(np.concatenate([by_pid[p] for p in pick])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(flat.mean()), float(lo), float(hi), int(len(flat)), len(pids)


PAIRS = (("midlayer_attn_lp_ft", "attn_ft"), ("midlayer_attn_lp_ft", "attention_probe"),
         ("midlayer_attn_lp_ft", "attn_lp_ft"), ("attention_probe", "attn_ft"),
         ("attn_lp_ft", "attn_ft"))
PAIRED_CSV = RESULTS / "ood_paired_clustered.csv"


def clustered_paired(a: dict, b: dict, pid_of: dict, rng) -> dict:
    """Mean of (a - b) over the phases both scored, resampling PARTICIPANTS."""
    common = [k for k in a if k in b and k in pid_of
              and np.isfinite(a[k]) and np.isfinite(b[k])]
    by_pid: dict = {}
    for k in common:
        by_pid.setdefault(pid_of[k], []).append(a[k] - b[k])
    pids = sorted(by_pid)
    if not pids:
        return {"diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan,
                "n_phases": 0, "n_participants": 0}
    flat = np.concatenate([by_pid[p] for p in pids])
    draws = [np.mean(np.concatenate([by_pid[p] for p in rng.choice(pids, len(pids))]))
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(flat.mean()), "diff_lo": float(lo), "diff_hi": float(hi),
            "n_phases": int(len(flat)), "n_participants": len(pids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", default=["gemma2", "llama"])
    args = ap.parse_args()
    annotations = load_annotations(required=True)
    from src.stage3.ood_detection import load_phases
    from src.stage3.prepare_conversations import CORPORA

    phases, _ = load_phases(list(CORPORA))
    phases_text = phases.set_index("phase_id")["text"].to_dict()
    rng = np.random.default_rng(SEED)
    rows = []
    for model in [resolve_model(m) for m in args.model]:
        for method, frame in phase_tables(model).items():
            pid_of = dict(zip(frame["phase_id"], frame["nickname"]))
            flagged = dict(zip(frame["phase_id"], frame["flagged"].astype(bool)))
            annotated = set(annotations["phase_id"])
            neg = frame[~frame["phase_id"].isin(annotated)]
            n_neg_flagged = int(neg["flagged"].astype(bool).sum())
            for view in VIEWS:
                sup = supervision(annotations, view)
                y = sup["label"].to_numpy(dtype=int)
                f = np.array([flagged.get(p, False) for p in sup["phase_id"]])
                tp, fn = int((f & (y == 1)).sum()), int((~f & (y == 1)).sum())
                tn, fp = int((~f & (y == 0)).sum()), int((f & (y == 0)).sum())
                ev_auprc, ev_margin = phase_evidence(method, model, annotations,
                                                     view, phases_text)
                a = clustered_mean(ev_auprc, pid_of, rng)
                m = clustered_mean(ev_margin, pid_of, rng)
                rows.append({
                    "method": method, "model": model, "annotator": view,
                    "auprc": a[0], "auprc_lo_cluster": a[1], "auprc_hi_cluster": a[2],
                    "auprc_n_phases": a[3], "auprc_n_participants": a[4],
                    "aopc": m[0], "aopc_lo_cluster": m[1], "aopc_hi_cluster": m[2],
                    "aopc_n_phases": m[3], "aopc_n_participants": m[4],
                    "n_pos": tp + fn, "n_neg_annotated": tn + fp,
                    "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                    "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                    "n_flagged_annotated": tp + fp,
                    "n_screen_negative": int(len(neg)),
                    "n_screen_negative_flagged": n_neg_flagged,
                    "threshold": float(frame["threshold"].iloc[0]),
                })
                r = rows[-1]
                print(f"  {model} {method:<22} {view:<10} AUPRC {r['auprc']:.3f} "
                      f"[{r['auprc_lo_cluster']:.3f}, {r['auprc_hi_cluster']:.3f}]  "
                      f"ΔAOPC {r['aopc']:.2f} [{r['aopc_lo_cluster']:.2f}, "
                      f"{r['aopc_hi_cluster']:.2f}]  sens {r['sensitivity']:.2f} "
                      f"spec {r['specificity']:.2f}  screen-neg flagged "
                      f"{n_neg_flagged}/{len(neg)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(table)} rows)")
    # paired, participant-clustered differences between methods
    prow = []
    for model in [resolve_model(m) for m in args.model]:
        tables = phase_tables(model)
        for a, b in PAIRS:
            if a not in tables or b not in tables:
                continue
            pid_of = dict(zip(tables[a]["phase_id"], tables[a]["nickname"]))
            for view in VIEWS:
                ea, ma = phase_evidence(a, model, annotations, view, phases_text)
                eb, mb = phase_evidence(b, model, annotations, view, phases_text)
                for metric, (xa, xb) in (("word_auprc", (ea, eb)), ("aopc", (ma, mb))):
                    r = clustered_paired(xa, xb, pid_of, rng)
                    prow.append({"model": model, "method_a": a, "method_b": b,
                                 "annotator": view, "metric": metric, **r})
                    print(f"  {model} {metric:<10} {a} - {b} [{view}]: {r['diff']:+.3f} "
                          f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] n={r['n_phases']}")
    pd.DataFrame(prow).to_csv(PAIRED_CSV, index=False)
    print(f"wrote {PAIRED_CSV} ({len(prow)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
