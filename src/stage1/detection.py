"""Detection AUROC for every trained method -> results/detection.csv.

How to run (CPU, reads what src.stage1.train wrote). --method defaults to
every method and --model defaults to gemma2:
    uv run python -m src.stage1.detection                      # all methods, gemma2
    uv run python -m src.stage1.detection --model llama        # the family control
    uv run python -m src.stage1.detection --method sae_probe
    uv run python -m src.stage1.detection --method full_ft lp_ft
    uv run python -m src.stage1.detection --method glove_bigru mentalbert   # no --model needed
    uv run python -m src.stage1.detection --method prompting

Rows for glove_bigru and mentalbert are backbone-free: they ignore --model and
label themselves glove-300d / mentalbert.

EVERY ROW IS SCORED ON THE SAME POSTS. The shared test set is the union of
the five study folds (SHARED_FOLDS), 1,761 posts, and each row keeps only its
method's out-of-fold scores on those posts. This is not cosmetic: the methods
did not all score the same corpus slice. Probes, the FT arms and mentalbert
scored these five folds; glove_bigru and prompting scored all ten; the anchor
scored fold 0 alone -- the easiest fold in the corpus -- and topped the table
on it. A method missing any shared-test post is an error here, not a quietly
smaller n.

One row per (method, model): a single AUROC over the shared test set's pooled
out-of-fold scores (never a mean of fold AUROCs), a 2,000-draw post-bootstrap
95% CI, three decision columns -- `accuracy`, `fpr`, `fnr` -- all read at the
threshold the method fitted on its training folds, and `fpr_at_tpr90`, which
is read off the ROC curve at a matched 90% sensitivity and is the only
false-positive number comparable across methods that chose different
thresholds. Rows are sorted by `auroc`, the one column that is neither
thresholded nor prevalence-dependent. src/stage1/METRICS.md says which
annotation rule each column answers to. Rows are upserted into
results/detection.csv, so the table accumulates across invocations. Every method in the study publishes exactly
one row per (method, model), so there is no `variant` column here; a method
that split its rows -- the way `sae_probe` would if it published one pooling
per row -- is refused by `_load_probe` rather than collapsed onto one key.

Score sources (results/detection/, written by src.stage1.train):
    probes           detect_{method}_{model}.parquet — carries `pred` and the
                     `threshold` behind it alongside `score`
    FT arms          detect_{method}_{model}_fold{k}.parquet — smoke cells are
                     filtered on the step count in the matching recipe sidecar,
                     results/train/train_{method}_{model}_fold{k}.json
    mentalbert       detect_mentalbert_mentalbert.parquet
    glove_bigru      detect_glove_{model}.parquet
    prompting        detect_prompt_{model}.csv
"""

import argparse
import glob as globlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.common import (ALL_METHODS, ARM_OF, EXPLAINER_BASE, artifact_path,
                              add_method_args, group_of, is_api_model,
                              resolve_model, selected_methods)
from src.utils.common import (TRAIN_DIR, RESULTS, ci, load_config,
                              metric_distribution, shared_draws,
                              DEFAULT_MODEL, MENTALBERT_ENCODER)
from src.utils.data_prep import load_corpus

OUT_CSV = RESULTS / "detection.csv"

# The five study folds every trained method holds out (src.stage1.train's
# DEFAULT_FOLDS). Their union is the shared test set this table is scored on.
SHARED_FOLDS = (0, 2, 4, 6, 8)

# The published columns, in order. Anything else an older version of this file
# wrote -- n_posts, n_folds, f1, tpr, tnr, precision, fpr_at_tpr90 and the raw
# tp/tn/fp/fn cells -- is dropped on the next write. n_posts and n_folds are
# gone because the shared test set makes them constant; f1 is gone because it
# charges a false positive and a false negative the same amount, which is the
# one thing this construct does not do. src/stage1/METRICS.md is the reference
# for what each column means.
COLUMNS = ["method", "group", "model", "train_post", "test_post",
           "auroc", "auroc_lo", "auroc_hi",
           "accuracy", "fpr", "fnr", "fpr_at_tpr90"]

# The sensitivity `fpr_at_tpr90` is read at. Every other decision column is
# read at each method's own fitted cut, and those cuts are not the same place
# on each method's curve -- so this is the one column on which a low false
# positive count means the method rejects absent posts, rather than that it
# says "present" less often.
TPR_TARGET = 0.90

# The column rows are sorted on. `auroc` is threshold-free and invariant to
# the class mix, so it is the only column whose order is a ranking of methods
# rather than of methods and their fitted cuts together. Sorting on `accuracy`
# instead moves `mlp_probe` nine places, all of it threshold -- see
# src/stage1/METRICS.md §5.
SORT_COLUMN = "auroc"

# `anchor` is not published here. It is a fold-0-only cell, and full_ft is the
# same full fine-tune over all five study folds; Stage 2 and Stage 3 still
# carry anchor rows of their own.
DETECTION_METHODS = tuple(m for m in ALL_METHODS if m != "anchor")

N_BOOT = 2000
SEED = 0


def shared_test() -> tuple[set[str], int]:
    """(the shared test post ids, the mean per-fold training-pool size).

    Every method in this table is cross-validated on the same stratified
    10-fold split (seed 42, pinned in data/lonxplain/posts.csv), so a fold's
    training pool is every post outside it -- 3,169 for fold 0 and 3,170 for
    the rest. `train_post` publishes the mean of those over SHARED_FOLDS
    rather than a per-fold column, since the folds differ by at most one post.

    Note what train_post is NOT: the folds' training pools overlap, so it is
    not a set disjoint from the test column. No post is ever scored by a model
    that trained on it -- that is what out-of-fold means -- but across the five
    folds every post trains something. Globally disjoint train and test sets
    would take a single fixed split, not cross-validation.
    """
    posts = load_corpus("lonxplain")[0]
    ids = set(posts[posts["fold"].isin(SHARED_FOLDS)]["id"])
    sizes = [len(posts) - int((posts["fold"] == k).sum()) for k in SHARED_FOLDS]
    return ids, round(sum(sizes) / len(sizes))


def _is_study_cell(path) -> bool:
    """False for smoke/debug fine-tune cells, which must never reach a table.

    Smoke runs land in a smoke/ subdirectory and never match this module's glob;
    the step-count check below is a second line of defence in case a smoke cell
    is ever copied beside real cells (full_finetune.min_steps).

    A cell's scores and its recipe sidecar share one stem under two stage
    prefixes (src/utils/common.py), so the sidecar is this file's name with
    `detect` swapped for `train`, in the train directory -- no name is parsed
    back into arm and model.
    """
    cell = Path(path)
    sidecar = (TRAIN_DIR
               / cell.name.replace("detect_", "train_", 1)).with_suffix(".json")
    if not sidecar.exists():
        return False
    with open(sidecar) as fh:
        info = json.load(fh)
    return int(info.get("steps", 0)) >= load_config()["full_finetune"]["min_steps"]


# --------------------------------------------------------------- score loaders
# Each returns (frame with columns [id, fold, label, score, pred?], model
# label) or None when the method has not been trained yet. `id` is what
# method_row restricts to the shared test set with, so every loader carries it.


def _load_probe(method: str, model: str):
    path = artifact_path("detect", method, model)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # src.stage1.train writes a `variant` column so a method that fits more
    # than one variant against the same activations keeps them apart. This
    # table publishes one row per (method, model) and has no column to carry
    # that split, so a score file that carries one is an error here rather
    # than a silent pooling of rows that were never meant to be pooled.
    if "variant" in df.columns:
        split = sorted(set(df["variant"].fillna("")) - {""})
        if split:
            raise SystemExit(
                f"{method} wrote variants {split} to {path.name}; "
                "detection.csv publishes one row per (method, model) and "
                "cannot represent them")
    if df.empty:
        return None
    # `pred` is the probe's decision at the threshold src.stage1.train fitted
    # on the training folds. A score file written before probes carried one has
    # no such column; it keeps the ranking and leaves the decision cells empty
    # rather than being re-thresholded here, where the only labels in reach are
    # the held-out ones.
    cols = ["id", "fold", "label", "score"] + (["pred"] if "pred" in df else [])
    return df[cols], model


def _load_arm(method: str, model: str):
    arm = ARM_OF[method]
    files = sorted(globlib.glob(
        str(artifact_path("detect", arm, model, fold="*"))))
    files = [f for f in files if _is_study_cell(f)]
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "role" in df:
        df = df[df["role"] == "in_domain"]
    df = df.rename(columns={"post_score": "score", "post_pred": "pred"})
    return df[["id", "fold", "label", "score", "pred"]], model


def _load_mentalbert(method: str, model: str):
    # Both backbone-free arms ignore --model: mentalbert is filed under the
    # encoder it fine-tunes, glove_bigru under the study default, so each is
    # read with the same arguments src.stage1.train wrote it with.
    path = artifact_path("detect", "mentalbert", MENTALBERT_ENCODER)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.rename(columns={"post_score": "score", "post_pred": "pred"})
    return df[["id", "fold", "label", "score", "pred"]], "mentalbert"


def _load_glove(method: str, model: str):
    path = artifact_path("detect", "glove_bigru", DEFAULT_MODEL)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.rename(columns={"post_score": "score", "post_pred": "pred"})
    return df[["id", "fold", "label", "score", "pred"]], "glove-300d"


def _load_prompting(method: str, model: str):
    path = artifact_path("detect", "prompting", model, ext="csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # NaN pred_label is the writer's marker for a response that did not parse
    # (src.stage1.train counts them as n_unparsed). Keep it missing rather than
    # letting astype(str) turn it into "nan" -> a hard "absent" prediction,
    # which would silently score as a true negative or a false negative.
    raw = df["pred_label"].astype("string").str.strip().str.lower()
    pred = pd.Series(
        np.where(raw.isin(("yes", "present")), "present", "absent"),
        index=df.index, dtype="object",
    ).mask(raw.isna())
    out = pd.DataFrame({
        "id": df["id"], "fold": df["fold"], "label": df["label"],
        "score": pd.to_numeric(df["present_score"], errors="coerce"),
        "pred": pred,
    })
    return out, model


def _load_explainer(method: str, model: str):
    # A post-hoc explainer is its base method's classifier with a different
    # word ranking, so its detection row IS the base's, read from the same
    # cells (arm or probe) and republished under the explainer's name.
    base = EXPLAINER_BASE[method]
    return _loader(base)(base, model)


def _loader(method: str):
    if method in EXPLAINER_BASE:
        return _load_explainer
    if method in ARM_OF:
        return _load_arm
    return {
        "mentalbert": _load_mentalbert,
        "glove_bigru": _load_glove,
        "prompting": _load_prompting,
    }.get(method, _load_probe)


# --------------------------------------------------------------------- metrics


CONFUSION_CELLS = ("tp", "tn", "fp", "fn")


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    """The four cells the decision columns are built from.

    Not published as counts -- they are corpus-size-dependent -- and not
    folded into an F1, which charges a false positive and a false negative the
    same amount. Three of the four annotation rules (the appraisal rule and
    the conflict and depression exclusions, configs/prompts.yaml) exist to
    hold back a "present" call on text that merely looks lonely, so which way
    a method errs is the clinically load-bearing part and a single symmetric
    summary hides it.

    tn is the cell no present-class summary reads: precision and recall are
    both blind to it, and it is what a false-positive rate is measured
    against.
    """
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def _fpr_at_tpr(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """The lowest false-positive rate at which this ranking reaches `target`
    sensitivity.

    Threshold-free, and therefore the one false-positive number two rows may
    be compared on: it asks every method the same question -- hold 90% of the
    truly-lonely posts, and how many non-lonely posts do you drag in with them
    -- instead of reporting where each method's own fitted cut happened to
    land. Read off the test ROC curve, so it is a property of the ranking and
    a diagnostic; it is not an operating point a deployment could have picked
    in advance without a calibration set.
    """
    fpr, tpr, _ = roc_curve(y, score)          # tpr is non-decreasing
    idx = int(np.searchsorted(tpr, target, side="left"))
    return float(fpr[idx]) if idx < len(fpr) else float("nan")


RATE_CELLS = ("accuracy", "fpr", "fnr")


def _rates(counts: dict[str, int]) -> dict[str, float]:
    """The decision cells, at the threshold the method fitted.

    accuracy = (tp + tn) / n: the share of posts called correctly, either
    way round. The one column here that is not conditioned on the true
    class, so it is the one that moves when the class mix does -- on this
    54.7%-present corpus a method that answers "present" to everything
    scores 0.547, which is the floor to read it against rather than 0.5.
    It also folds the two errors into one number at a weight the construct
    does not endorse (three of the four annotation rules hold down the
    false positive alone), which is why `fpr` and `fnr` are published
    unfolded beside it.

    fpr = fp / (fp + tn): of the posts that are truly absent, the share this
    method flagged anyway -- the quantity the appraisal rule and the conflict
    and depression exclusions exist to hold down, and the column this study
    is built around.
    fnr = fn / (fn + tp): of the posts that are truly present, the share it
    missed -- published beside fpr because fpr alone is half a metric. A
    method that never says present scores a perfect 0.0 fpr and an fnr of
    1.0, and violates the temporal rule on every post. Both columns are
    errors and both read lower-is-better, which is the point of publishing
    the miss rate rather than its complement.
    fpr and fnr are conditioned on the true class, so neither moves when the
    mix of present and absent posts changes. Accuracy does move, and this
    corpus is 54.7% present, far denser than any screening population, so it
    flatters a method in a way the two rates do not.
    """
    neg = counts["fp"] + counts["tn"]
    pos = counts["fn"] + counts["tp"]
    total = neg + pos
    return {
        "accuracy": (counts["tp"] + counts["tn"]) / total if total else float("nan"),
        "fpr": counts["fp"] / neg if neg else float("nan"),
        "fnr": counts["fn"] / pos if pos else float("nan"),
    }


def method_row(method: str, model: str, test_ids: set[str],
               train_post: int) -> dict | None:
    loaded = _loader(method)(method, model)
    if loaded is None:
        print(f"  {method:<20} not trained yet — run src.stage1.train first")
        return None
    df, model_label = loaded

    # The shared test set, enforced. glove_bigru and prompting score all ten
    # folds, so the restriction actually drops rows for them; for everything
    # else it is a no-op that proves the coverage. A method that is MISSING
    # shared-test posts is refused: silently scoring such a row on whatever
    # subset it happens to have is exactly the bug this table had.
    have = set(df["id"])
    missing = test_ids - have
    if missing:
        raise SystemExit(
            f"{method} ({model_label}) has no score for {len(missing)} of the "
            f"{len(test_ids)} shared-test posts (folds {list(SHARED_FOLDS)}); "
            f"train it on those folds or drop it from this table")
    df = df[df["id"].isin(test_ids)]

    # Unparsed prompting responses (NaN score, NaN pred) stay in the frame:
    # they are dropped from AUROC, which needs a number to rank, and counted
    # as a hard "absent" call in the decision columns, where a method that
    # returned no usable answer has still failed to flag the post. Every row
    # therefore decides on all len(test_ids) posts.
    y = (df["label"] == "present").to_numpy().astype(int)
    ranked = df[np.isfinite(df["score"].to_numpy(dtype=float))]
    auroc = lo = hi = fpr90 = float("nan")
    if len(ranked):
        y_ranked = (ranked["label"] == "present").to_numpy().astype(int)
        s_ranked = ranked["score"].to_numpy(dtype=float)
        draws = shared_draws(len(ranked), N_BOOT, SEED)
        lo, hi = ci(metric_distribution(roc_auc_score, y_ranked, s_ranked,
                                        draws))
        auroc = float(roc_auc_score(y_ranked, s_ranked))
        fpr90 = _fpr_at_tpr(y_ranked, s_ranked, TPR_TARGET)
    # A method that carries no decision publishes empty cells, rather than
    # having a cut invented for it at 0.5 here.
    counts = dict.fromkeys(CONFUSION_CELLS, float("nan"))
    rates = dict.fromkeys(RATE_CELLS, float("nan"))
    if "pred" in df and df["pred"].notna().any():
        counts = _confusion(y, (df["pred"] == "present").to_numpy().astype(int))
        rates = _rates(counts)
    row = {
        "method": method,
        "group": group_of(method),
        "model": model_label,
        "train_post": train_post if method != "prompting" else 0,
        "test_post": int(len(df)),
        "auroc": auroc,
        "auroc_lo": lo,
        "auroc_hi": hi,
        **rates,
        "fpr_at_tpr90": fpr90,
    }
    decided = not np.isnan(rates["fpr"])
    rate_cell = (f"acc {rates['accuracy']:.3f}  fpr {rates['fpr']:.3f}  "
                 f"fnr {rates['fnr']:.3f}" if decided
                 else "acc —  fpr —  fnr —")
    auroc_cell = ("—" if np.isnan(auroc)
                  else f"{auroc:.3f} [{lo:.3f}, {hi:.3f}]")
    fpr90_cell = "—" if np.isnan(fpr90) else f"{fpr90:.3f}"
    print(f"  {method:<16} {model_label:<12} AUROC {auroc_cell}  "
          f"{rate_cell}  fpr@tpr90 {fpr90_cell}  ({row['test_post']} posts)")
    return row


def upsert(rows: list[dict]):
    table = pd.DataFrame(rows)
    if OUT_CSV.exists():
        # Stale columns from an older version of this file -- `variant`,
        # n_posts, n_folds, f1, fnr -- are dropped by the reindex below rather
        # than carried forward.
        old = pd.read_csv(OUT_CSV)
        # A hand-edit to the published table lands a non-numeric cell in a
        # metric column, and the only symptom is a str-vs-float TypeError out
        # of the sort below. Name the cell instead of coercing it: a metric
        # that will not parse is a corrupted result, not a missing one.
        for col in COLUMNS[5:]:
            if col not in old.columns:
                continue
            bad = old[col][pd.to_numeric(old[col], errors="coerce").isna()
                           & old[col].notna()]
            for i, cell in bad.items():
                raise SystemExit(
                    f"{OUT_CSV}: row {i + 2} has a non-numeric {col} "
                    f"({cell!r}); repair it before rewriting the table")
        # Rows this table no longer publishes, whatever is being recomputed
        # now: the fold-0-only anchor, and any decision-only API backbone. An
        # API row would also be sorted on an `accuracy` it has no scores
        # behind, beside rows measured on the shared test set -- the
        # comparison this file exists to make impossible.
        retired = old["method"].eq("anchor") | old["model"].map(is_api_model)
        if retired.any():
            for _, r in old[retired].iterrows():
                print(f"  dropping retired row {r['method']} / {r['model']}")
            old = old[~retired]
        keys = set(zip(table["method"], table["model"]))
        old = old[~old.apply(
            lambda r: (r["method"], r["model"]) in keys, axis=1)]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(SORT_COLUMN, ascending=False)
    table = table.reindex(columns=COLUMNS)
    # Post counts are counts. Read back through a CSV that predates a column
    # they land as floats ("3170.0"), and one upsert is enough to make the
    # published table carry them that way forever.
    for col in ("train_post", "test_post"):
        table[col] = table[col].astype("Int64")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(RESULTS.parent)} ({len(table)} rows)")


def main():
    ap = argparse.ArgumentParser(
        description="Detection metrics per method -> results/detection.csv")
    add_method_args(ap, choices=DETECTION_METHODS)
    args = ap.parse_args()
    methods = selected_methods(args, choices=DETECTION_METHODS)
    model = resolve_model(args.model)
    if is_api_model(model):
        # An API backbone returns a label and no token distribution, so it can
        # publish neither an AUROC nor the matched-sensitivity column every
        # other row is compared on. A decision-only row cannot be ranked
        # against ranked ones, so this table does not carry one.
        raise SystemExit(
            f"--model {model} returns decisions and no scores, so it has "
            f"neither an auroc nor an fpr_at_tpr90 to set beside the "
            f"decision columns; this table publishes local backbones only")

    test_ids, train_post = shared_test()
    print(f"detection — model {model}; shared test set {len(test_ids)} posts "
          f"(folds {list(SHARED_FOLDS)}), {train_post} training posts per fold")
    rows = [r for m in methods
            if (r := method_row(m, model, test_ids, train_post)) is not None]
    if not rows:
        raise SystemExit("no method has scores on disk yet; run src.stage1.train")
    upsert(rows)


if __name__ == "__main__":
    main()
