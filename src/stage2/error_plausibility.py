"""Do the arms that detect as well as Full FT explain any better?
-> results/error_plausibility.csv

Stage 1 says which methods reach Full FT's AUROC. Stage 2 says how well each
one's own token attribution matches a psychiatrist's span. This CLI joins the
two AT THE POST LEVEL for the methods that clear the detection bar, and splits
each method's plausibility by what it did with the post:

    precision          tp / (tp + fp) at its OWN fitted threshold -- of the
                       posts it flags, the share that are really present
    precision_at_tpr90 the same, with every method first held at 90%
                       sensitivity, which is the version that may RANK methods
    fpr, fnr           its two error rates at its own fitted threshold
    fpr_at_tpr90       the false-alarm rate at matched sensitivity
    tp, fp, fn, tn     the counts all of the above are computed from
    word_auprc         its word AUPRC over the span-eval posts (all of them)
    word_auprc_caught  the same, over the span-eval posts it called `present`
    word_auprc_missed  the same, over the span-eval posts it called `absent`

WHY THERE IS NO `word_auprc_false_positive` COLUMN
--------------------------------------------------
A false positive is a post the psychiatrist marked `absent`, and an `absent`
post carries no annotated span: `span_start` is -1 and no word is gold. AUPRC
is the area under a precision-recall curve, which does not exist for a ranking
with no positive class -- `src.utils.dla._word_plausibility` returns NaN for
exactly this reason, and filling one in would be inventing a measurement. The
false-positive RATE is therefore reported and its plausibility is not, and no
amount of joining changes that: the annotation the metric scores against was
never made. `plot/plot_feature_distortion.py` shows what a false positive was
decided on instead, one post at a time, which is the only form that evidence
can take.

WHICH FALSE-POSITIVE COLUMN TO READ. Both are published because they answer
different questions, and only one of them is a fair comparison. Every method
is thresholded by the same RULE -- `best_f1_threshold` on its training folds,
out of fold for every row -- but the same rule lands different methods at
different places on their own ROC curves, so `fpr` is the cost of the
operating point a method actually shipped and NOT a ranking. Two methods with
identical AUROC can differ in `fpr` purely by being more or less conservative:
`mlp_probe` has the lowest `fpr` in the whole study (0.132) and the highest
miss rate with it (0.345). `fpr_at_tpr90` holds every method at 90%
sensitivity first and then reads the false-alarm rate, which is the
comparison. src/stage1/METRICS.md says the same thing in more detail; this
CLI carries both columns so a figure cannot accidentally rank on the wrong
one.

THE SPLIT IS THE POINT. `word_auprc` alone confounds two things a reader will
want separated: an arm could look plausible only on the posts it gets right,
or it could rank the annotated span badly everywhere. Splitting on the arm's
OWN decision answers it, and `word_auprc_caught - word_auprc_missed` is the
size of the confound.

WHICH METHODS. Every method whose AUROC point estimate reaches the lower
confidence bound of `--reference` (default full_ft) -- "close to or better
than Full FT" made into a rule rather than an eyeball. `--all-methods` drops
the gate.

How to run:
    uv run python -m src.stage2.error_plausibility                  # gemma2
    uv run python -m src.stage2.error_plausibility --model llama
    uv run python -m src.stage2.error_plausibility --all-methods

Per-post word AUPRC comes from wherever that method already keeps it: the
fine-tuning arms from the `plaus_*` cells their checkpoints wrote (no model is
loaded), the frozen probes by refitting out of fold on the cached activations,
on CPU, exactly as `src.stage2.dla_auprc` does. Nothing here trains, and
nothing here needs a GPU.

That last sentence is enforced rather than asserted: `main` hides every card
before torch is imported (`pin_visible_devices(..., need_gpu=False)`), which is
what `src.stage2.dla_auprc` does for the same refits. Without it the probe
fitters silently take whatever GPU the driver enumerates first and die on a
busy one -- llama31-8b's layer-24 activations at `attention_mlp_probe`'s pooling
step ask for ~700MB on a card a neighbour already owns. `--gpu` opts back in.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.utils.common import (
    ARM_OF,
    CELLS_FROM,
    EXPLAINER_BASE,
    add_device_args,
    DENSE_PROBES,
    DENSE_PROBE_LAYER,
    RESULTS,
    artifact_path,
    load_config,
    mean_ci,
    method_label,
    min_free_gb_for,
    model_label,
    pin_visible_devices,
    resolve_model,
)

OUT_CSV = RESULTS / "error_plausibility.csv"

# The method the gate is measured against. Full FT is the study's standard
# recipe and its worst plausibility cell, which is what makes "as accurate as
# Full FT" the interesting bar to hold the other arms to.
REFERENCE = "full_ft"


# ------------------------------------------------------------- the two joins


def per_post_predictions(method: str, model: str) -> pd.DataFrame:
    """`id -> label, score, pred` for one method's held-out decisions.

    The two families spell the same thing differently on disk -- the arms write
    `post_score`/`post_pred` in per-fold cells with a `corpus` column, the
    probes write `score`/`pred` in one file -- so the difference is absorbed
    here and every caller downstream sees `id`, `label`, `score`, `pred`.
    """
    # An explainer decides with its base method's classifier, so its
    # decisions are that method's cells; only its word ranking is its own.
    base = EXPLAINER_BASE.get(method, method)
    if base in ARM_OF:
        frame = _concat(artifact_path("detect", ARM_OF[base], model, fold="*"))
        frame = frame[frame["corpus"] == "lonxplain"]
        frame = frame.rename(columns={"post_pred": "pred",
                                      "post_score": "score"})
    else:
        frame = _concat(artifact_path("detect", base, model))
    return frame[["id", "label", "score", "pred"]].drop_duplicates("id")


def _counts(is_present: np.ndarray, flagged: np.ndarray) -> dict:
    return {
        "tp": int((flagged & is_present).sum()),
        "fp": int((flagged & ~is_present).sum()),
        "fn": int((~flagged & is_present).sum()),
        "tn": int((~flagged & ~is_present).sum()),
    }


def _precision(counts: dict) -> float:
    flagged = counts["tp"] + counts["fp"]
    return counts["tp"] / flagged if flagged else float("nan")


def matched_sensitivity(decisions: pd.DataFrame, target: float = 0.90) -> dict:
    """Counts with the method first held at `target` recall.

    Every method here is thresholded by the same RULE, but the same rule lands
    different methods at different points on their own ROC curves, so a
    precision read at each method's own threshold is the precision of the
    operating point it shipped and not a ranking: a more conservative method
    buys precision with recall. Sliding every method to the same recall first
    is what makes the column comparable.

    The cut is taken at a SCORE, not at a rank, so posts tied on the score all
    fall on the same side of it -- ranking through a tie would hand one method
    a fractional post its scores cannot actually separate.
    """
    is_present = (decisions["label"] == "present").to_numpy()
    score = decisions["score"].to_numpy(dtype=float)
    n_present = int(is_present.sum())
    if not n_present:
        raise SystemExit("no present posts in the held-out set")
    order = np.argsort(-score, kind="mergesort")
    reached = np.cumsum(is_present[order])
    need = int(np.ceil(target * n_present))
    rank = int(np.searchsorted(reached, need, side="left"))
    if rank >= len(order):
        raise SystemExit(f"no threshold reaches {target:.0%} sensitivity")
    counts = _counts(is_present, score >= score[order][rank])
    counts["recall"] = counts["tp"] / n_present
    return counts


def per_post_word_auprc(method: str, model: str, folds, seed: int,
                        reduction: str) -> pd.DataFrame:
    """`id -> word_auprc, chance` on the span-eval posts, for one method."""
    if method in ARM_OF or method in CELLS_FROM:
        return _arm_word_auprc(method, model)
    if method in DENSE_PROBES:
        return _probe_word_auprc(method, model, folds, seed, reduction)
    raise SystemExit(
        f"{method} keeps no per-post word AUPRC this CLI can read. The "
        "fine-tuning arms write `plaus_*` cells and the dense probes are "
        "refitted here; an SAE probe would need its dictionary loaded, which "
        "this CLI deliberately does not do.")


def _arm_word_auprc(method: str, model: str) -> pd.DataFrame:
    """The per-post cells the published `word_auprc` row is the mean of."""
    frame = _concat(artifact_path("plaus", ARM_OF.get(method, method), model,
                                  fold="*"))
    return frame.rename(columns={"word_gold_base_rate": "chance"})[
        ["id", "word_auprc", "chance"]]


def _probe_word_auprc(method: str, model: str, folds, seed: int,
                      reduction: str) -> pd.DataFrame:
    """Refit the probe out of fold and score its own token attribution.

    Deliberately the same calls `src.utils.dla.dense_probe_rows` makes -- the
    same context, the same fitters, the same `post_word_auprc` -- so a probe's
    rows here and its published row in results/dla_auprc.csv are the same
    measurement, differing only in that this one is kept per post instead of
    averaged.
    """
    from src import probe as probing
    from src.utils.dla import _span_eval

    from src.utils.common import probe_layer_of

    cfg = load_config()
    layer = probe_layer_of(method, model)
    print(f"  refitting {method} on {model} layer {layer} activations (CPU)",
          flush=True)
    ctx = probing.build_context(model, layer, cfg, seed)
    by_fold, plaus_ids = _span_eval(ctx.posts, cfg, folds)
    pos = {pid: i for i, pid in enumerate(plaus_ids)}
    _, targets, _, word_base, _ = probing.span_structures(
        ctx.posts, lambda pid: ctx.acts[pid]["offsets"], plaus_ids)

    scores = np.full(len(plaus_ids), np.nan)
    for fold in folds:
        fitted = probing.FITTERS[method](probing.for_fold(ctx, fold))
        if fitted.token_scores is None:
            raise SystemExit(
                f"{method}: {fitted.token_source} — it has no token "
                "attribution, so it has no plausibility to split")
        for pid in by_fold[fold]:
            scores[pos[pid]] = probing.post_word_auprc(
                np.asarray(fitted.token_scores(ctx.index_of[pid]),
                           dtype=np.float64),
                targets[pid], reduction)
        print(f"    fold {fold} scored", flush=True)
    return pd.DataFrame({"id": plaus_ids, "word_auprc": scores,
                         "chance": word_base})


def _concat(pattern) -> pd.DataFrame:
    import glob as globlib

    paths = sorted(globlib.glob(str(pattern)))
    if not paths:
        raise SystemExit(f"no cells matching {pattern}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# -------------------------------------------------------------- the row


def gated_methods(detection: pd.DataFrame, reference: str,
                  all_methods: bool) -> list[str]:
    """Methods whose AUROC reaches `reference`'s lower confidence bound.

    A point estimate against a confidence bound, rather than two point
    estimates: "close to or better than Full FT" has to admit the arms that
    are not distinguishable from it, and the bound is where that stops being a
    judgement call.
    """
    if all_methods:
        return detection["method"].tolist()
    row = detection[detection["method"] == reference]
    if row.empty:
        raise SystemExit(f"{reference} has no row for this model in "
                         "results/detection.csv")
    bar = float(row["auroc_lo"].iloc[0])
    keep = detection[detection["auroc"] >= bar]
    print(f"gate: AUROC >= {bar:.4f}, the lower bound of {reference}'s "
          f"{float(row['auroc'].iloc[0]):.4f} — {len(keep)} of "
          f"{len(detection)} methods clear it")
    return keep["method"].tolist()


def row_for(method: str, model: str, detection: pd.DataFrame, folds, seed,
            reduction: str) -> dict | None:
    stage1 = detection[detection["method"] == method].iloc[0]
    try:
        auprc = per_post_word_auprc(method, model, folds, seed, reduction)
        preds = per_post_predictions(method, model)
    except SystemExit as exc:
        print(f"  {method}: {exc} — SKIPPED")
        return None

    joined = auprc.merge(preds, on="id", how="inner")
    if joined.empty:
        print(f"  {method}: no post is in both its decisions and its "
              "plausibility cells — SKIPPED")
        return None
    # Every span-eval post is a `present` post with a span, so on this subset
    # the arm's own decision splits them into the ones it caught and the ones
    # it missed. There is no third group here, and no false positives: a
    # false positive is by definition a post this subset excludes.
    if not (joined["label"] == "present").all():
        raise SystemExit(
            f"{method}: the span-eval join contains `absent` posts, which "
            "carry no gold word — the caught/missed split would be scoring "
            "AUPRCs that do not exist")
    caught = joined[joined["pred"] == "present"]["word_auprc"]
    missed = joined[joined["pred"] == "absent"]["word_auprc"]

    # The decision columns are computed over the whole held-out set, not over
    # the span-eval subset the AUPRC is measured on: precision is a property of
    # every call the method made, and restricting it to the annotated posts
    # would quote it over a set with no `absent` posts in it at all.
    own = _counts((preds["label"] == "present").to_numpy(),
                  (preds["pred"] == "present").to_numpy())
    matched = matched_sensitivity(preds)

    overall, lo, hi = mean_ci(joined["word_auprc"])
    c, c_lo, c_hi = mean_ci(caught)
    m, m_lo, m_hi = mean_ci(missed)
    return {
        "method": method,
        "group": stage1["group"],
        "model": model,
        "auroc": float(stage1["auroc"]),
        "auroc_lo": float(stage1["auroc_lo"]),
        "auroc_hi": float(stage1["auroc_hi"]),
        # The two error ratios, taken from the published Stage 1 table rather
        # than recomputed, so this CLI cannot disagree with it.
        "fpr": float(stage1["fpr"]),
        "fpr_at_tpr90": float(stage1["fpr_at_tpr90"]),
        "fnr": float(stage1["fnr"]),
        **own,
        "precision": _precision(own),
        "precision_at_tpr90": _precision(matched),
        "recall_at_tpr90": float(matched["recall"]),
        "word_auprc": overall, "word_auprc_lo": lo, "word_auprc_hi": hi,
        "word_auprc_caught": c, "word_auprc_caught_lo": c_lo,
        "word_auprc_caught_hi": c_hi,
        "word_auprc_missed": m, "word_auprc_missed_lo": m_lo,
        "word_auprc_missed_hi": m_hi,
        "n_posts": int(np.isfinite(joined["word_auprc"]).sum()),
        "n_caught": int(np.isfinite(caught).sum()),
        "n_missed": int(np.isfinite(missed).sum()),
        # What a random ranker scores on these posts. A word AUPRC means
        # nothing until chance on the same posts is beside it -- which is why
        # the mean runs over the posts the AUPRC ITSELF is defined on. The two
        # families disagree on which posts those are unless this is said
        # explicitly: the probe path already NaNs the base rate of a post with
        # no gold word or nothing but gold words, and the arm path records it
        # regardless, so an unrestricted mean would quote chance for the arms
        # over a wider post set than the AUPRC beside it was averaged over.
        "word_chance": float(np.nanmean(
            joined["chance"].where(np.isfinite(joined["word_auprc"])))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Error rates and split plausibility for the methods that "
                    "detect as well as Full FT -> results/error_plausibility.csv")
    ap.add_argument("--model", default="gemma2")
    ap.add_argument("--reference", default=REFERENCE,
                    help="the method the AUROC gate is measured against")
    ap.add_argument("--all-methods", action="store_true",
                    help="drop the AUROC gate and take every method that has "
                         "the cells")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--word-aggregation", default=None,
                    choices=("max", "mean", "sum"))
    ap.add_argument("--gpu", action="store_true",
                    help="let the probe refits use a GPU (default: pinned CPU)")
    add_device_args(ap)
    args = ap.parse_args()

    model = resolve_model(args.model)
    # Before anything imports torch, and before any probe is fitted: this CLI
    # reads cached activations and scored cells, so the default is no GPU at
    # all rather than whichever card the driver happens to hand over.
    pin_visible_devices(args.devices, need_gpu=bool(args.gpu),
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)
    cfg = load_config()
    reduction = (args.word_aggregation
                 or cfg["audit"]["word_score_aggregation"])

    detection = pd.read_csv(RESULTS / "detection.csv")
    detection = detection[detection["model"] == model]
    if detection.empty:
        raise SystemExit(f"results/detection.csv has no rows for {model}")
    methods = gated_methods(detection, args.reference, args.all_methods)

    rows = []
    for method in methods:
        print(f"{method}:", flush=True)
        row = row_for(method, model, detection, args.folds, args.seed,
                      reduction)
        if row is not None:
            rows.append(row)
            print(f"  precision {row['precision']:.3f} (at tpr90 "
                  f"{row['precision_at_tpr90']:.3f}, recall "
                  f"{row['recall_at_tpr90']:.3f})  fpr {row['fpr']:.3f}  "
                  f"word AUPRC {row['word_auprc']:.3f} "
                  f"(caught {row['word_auprc_caught']:.3f} on "
                  f"{row['n_caught']}, missed {row['word_auprc_missed']:.3f} "
                  f"on {row['n_missed']}), chance {row['word_chance']:.3f}")
    if not rows:
        raise SystemExit("no method produced a row — see the messages above")

    table = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        keys = set(zip(table["method"], table["model"]))
        old = old[~old.apply(lambda r: (r["method"], r["model"]) in keys,
                             axis=1)]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["model", "auroc"], ascending=[True, False])
    table.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.name} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
