"""Uncertainty beyond the published intervals -> results/per_fold_auroc.csv and
results/paired_bootstrap.csv.

How to run (CPU; reads the detection cells and the Stage 2 cells only):
    uv run python -m src.stage2.uncertainty
    uv run python -m src.stage2.uncertainty --model gemma2

TWO THINGS THE PUBLISHED TABLES DO NOT SAY
------------------------------------------
1. Per-fold AUROC. results/detection.csv is one AUROC over the pooled
   out-of-fold scores with a 2,000-draw post bootstrap. This writes, per
   method and backbone, the five held-fold AUROCs with their mean and SD, so
   the fold-to-fold spread can be quoted beside the pooled interval.

2. Paired bootstrap. The published intervals are per method; a difference
   between two methods needs the SAME resampled posts on both sides. For each
   named pair this draws 2,000 post resamples shared by both methods and
   reports the difference with its interval and a two-sided bootstrap p for
   three statistics: pooled AUROC (the shared test set), word AUPRC (the
   span-eval posts both methods scored) and ΔAOPC (the posts both methods
   published a margin on). A pair is reported only where both cells exist.

The pairs are the study's own contrasts: each attention arm against its
mean-pooled twin, each probe against the fine-tune at the same site, and the
post-hoc explainers against the decomposition they explain.
"""

from __future__ import annotations

import argparse
import glob as globlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.common import (ARM_OF, EXPLAINER_BASE, RESULTS, artifact_path,
                              group_of, resolve_model)

N_BOOT = 2000
SEED = 0
FOLDS = (0, 2, 4, 6, 8)
PAIRS = (
    # pooling pairs (the Δ column of the main table)
    ("attention_probe", "linear_probe"),
    ("attention_mlp_probe", "mlp_probe"),
    ("sae_attention_probe", "sae_probe"),
    ("attn_ft", "full_ft"),
    ("midlayer_attn_ft", "midlayer_ft"),
    ("attn_lp_ft", "lp_ft"),
    ("midlayer_attn_lp_ft", "midlayer_lp_ft"),
    ("attn_lora_ft", "lora_ft"),
    ("attn_span_ft", "span_ft"),
    # every attention-pooled row against the study's standard fine-tune
    # (plot/plot_paired_differences.py)
    ("midlayer_attn_ft", "attn_ft"),
    ("attn_lp_ft", "attn_ft"),
    ("midlayer_attn_lp_ft", "attn_ft"),
    ("attn_lora_ft", "attn_ft"),
    ("attn_span_ft", "attn_ft"),
    ("attention_probe", "attn_ft"),
    ("attention_mlp_probe", "attn_ft"),
    ("sae_probe", "attn_ft"),
    ("occl_attn_ft", "attn_ft"),
    ("gxi_ft", "full_ft"),
    # the proposed method against each of its neighbours (2026-10-06,
    # reviewer response): the structural contrast (vs LP-FT), the
    # representational contrast (vs ML-FT), and the frozen probe it starts from
    ("midlayer_attn_lp_ft", "attn_lp_ft"),
    ("midlayer_attn_lp_ft", "midlayer_attn_ft"),
    ("midlayer_attn_lp_ft", "attention_probe"),
    ("attn_lp_ft", "attention_probe"),
    # the final-layer frozen control against the mid-layer probe and Full FT
    ("final_attention_probe", "attention_probe"),
    ("final_attention_probe", "attn_ft"),
)


# ------------------------------------------------------------ per-post data


def post_scores(method: str, model: str) -> pd.DataFrame | None:
    """id, fold, y, score on the shared test folds."""
    base = EXPLAINER_BASE.get(method, method)
    if base in ARM_OF:
        files = sorted(globlib.glob(str(artifact_path("detect", ARM_OF[base], model,
                                                      fold="*"))))
        if not files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "role" in df:
            df = df[df["role"] == "in_domain"]
        if "corpus" in df:
            df = df[df["corpus"] == "lonxplain"]
        df = df.rename(columns={"post_score": "score"})
    else:
        path = artifact_path("detect", base, model)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "variant" in df:
            df = df[df["variant"].fillna("") == ""]
    df = df[df["fold"].isin(FOLDS)]
    out = pd.DataFrame({"id": df["id"], "fold": df["fold"],
                        "y": (df["label"] == "present").astype(int),
                        "score": pd.to_numeric(df["score"], errors="coerce")})
    return out.dropna(subset=["score"]).drop_duplicates("id")


def post_auprc(method: str, model: str) -> pd.Series | None:
    """id -> word AUPRC on the span-eval posts, from the plaus_* cells (arms and
    explainers) or by refitting the probe as dla_auprc does."""
    if method in ARM_OF or method in EXPLAINER_BASE:
        name = ARM_OF.get(method, method)
        files = sorted(globlib.glob(str(artifact_path("plaus", name, model,
                                                      fold="*"))))
        if not files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df.set_index("id")["word_auprc"]
    if group_of(method) == "probe":
        from src.stage2.error_plausibility import _probe_word_auprc
        from src.utils.common import DENSE_PROBES, load_config

        if method not in DENSE_PROBES:      # the SAE probes have no dense refit
            return None
        try:
            frame = _probe_word_auprc(method, model, list(FOLDS), SEED,
                                      load_config()["audit"]["word_score_aggregation"])
        except (Exception, SystemExit) as exc:   # a pair is skipped, not the run
            print(f"  {method}: no per-post AUPRC ({exc}); pair skipped")
            return None
        return frame.set_index("id")["word_auprc"]
    return None


def post_margin(method: str, model: str) -> pd.Series | None:
    """id -> ΔAOPC margin over the published posts (positive, non-small gap)."""
    if group_of(method) == "probe":
        path = artifact_path("delete", "probes", model)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        frame = frame[frame["method"] == method]
    else:
        files = sorted(globlib.glob(str(artifact_path("delete", ARM_OF[method], model,
                                                      fold="*"))))
        if not files:
            return None
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if frame.empty:
        return None
    d = frame[(frame["intervention"] == "word_delete")
              & (~frame["small_logit_gap"].astype(bool))
              & (frame["clean_empty_logit_gap"] > 0)]
    tgt = d[d["control"] == "target"].groupby("id")["normalized_aopc"].first()
    ctl = d[d["control"] == "random_words"].groupby("id")["normalized_aopc"].mean()
    common = tgt.index.intersection(ctl.index)
    return (tgt.loc[common] - ctl.loc[common]).dropna()


# ------------------------------------------------------------------ stats


def per_fold_auroc(scores: pd.DataFrame) -> dict:
    vals = []
    for fold in FOLDS:
        f = scores[scores["fold"] == fold]
        if f["y"].nunique() == 2:
            vals.append(roc_auc_score(f["y"], f["score"]))
    vals = np.asarray(vals, dtype=float)
    return {"n_folds": int(len(vals)), "fold_mean": float(vals.mean()),
            "fold_sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            **{f"fold{k}": (float(vals[i]) if i < len(vals) else float("nan"))
               for i, k in enumerate(FOLDS)}}


def paired(a: np.ndarray, b: np.ndarray, stat, rng) -> dict:
    """Difference stat(a) - stat(b) on shared resamples of the paired rows."""
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    point = stat(a) - stat(b)
    draws = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # two-sided bootstrap p: the share of draws on the other side of zero
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(point), "diff_lo": float(lo), "diff_hi": float(hi),
            "p_boot": float(min(1.0, p)), "n_paired": int(n)}


def paired_auroc(sa: pd.DataFrame, sb: pd.DataFrame, rng) -> dict:
    m = sa.merge(sb, on="id", suffixes=("_a", "_b"))
    y = m["y_a"].to_numpy(); xa = m["score_a"].to_numpy(); xb = m["score_b"].to_numpy()
    n = len(m)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    point = roc_auc_score(y, xa) - roc_auc_score(y, xb)
    draws = []
    for i in idx:
        yi = y[i]
        if yi.min() == yi.max():
            continue
        draws.append(roc_auc_score(yi, xa[i]) - roc_auc_score(yi, xb[i]))
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(point), "diff_lo": float(lo), "diff_hi": float(hi),
            "p_boot": float(min(1.0, p)), "n_paired": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", default=["gemma2", "llama"])
    args = ap.parse_args()
    models = [resolve_model(m) for m in args.model]
    rng = np.random.default_rng(SEED)

    methods = sorted({m for pair in PAIRS for m in pair})
    fold_rows, pair_rows = [], []
    for model in models:
        scores = {m: post_scores(m, model) for m in methods}
        for m, s in scores.items():
            if s is None:
                continue
            fold_rows.append({"method": m, "model": model, **per_fold_auroc(s)})
        auprc = {m: post_auprc(m, model) for m in methods}
        margin = {m: post_margin(m, model) for m in methods}
        for a, b in PAIRS:
            for metric, table, fn in (("auroc", scores, None),
                                      ("word_auprc", auprc, np.mean),
                                      ("aopc_margin", margin, np.mean)):
                xa, xb = table.get(a), table.get(b)
                if xa is None or xb is None:
                    continue
                if metric == "auroc":
                    row = paired_auroc(xa, xb, rng)
                else:
                    common = xa.dropna().index.intersection(xb.dropna().index)
                    if len(common) < 10:
                        continue
                    row = paired(xa.loc[common].to_numpy(dtype=float),
                                 xb.loc[common].to_numpy(dtype=float), fn, rng)
                pair_rows.append({"model": model, "method_a": a, "method_b": b,
                                  "metric": metric, **row})
                print(f"  {model} {metric:<12} {a} - {b}: {row['diff']:+.3f} "
                      f"[{row['diff_lo']:+.3f}, {row['diff_hi']:+.3f}] "
                      f"p={row['p_boot']:.3f} (n={row['n_paired']})")
    pd.DataFrame(fold_rows).to_csv(RESULTS / "per_fold_auroc.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(RESULTS / "paired_bootstrap.csv", index=False)
    print(f"wrote per_fold_auroc.csv ({len(fold_rows)} rows), "
          f"paired_bootstrap.csv ({len(pair_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
