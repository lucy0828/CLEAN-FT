"""Motivation figure: the words each method actually decides on.

How to run. Everything the default page needs is cached, so it redraws on the
CPU in seconds. Computing a missing cell is what needs a device: an arm loads
the backbone behind the study gate, a probe is fitted on the CPU.
    uv run python plot/plot_feature_distortion.py
    CUDA_VISIBLE_DEVICES=1 uv run python plot/plot_feature_distortion.py \
        --methods full_ft --allow-model-process     # a missing arm cell
    uv run python plot/plot_feature_distortion.py \
        --methods attention_probe                   # a missing probe fit

Writes:
    plot/figs/fig_feature_distortion.png

Two real LonXplain posts, and under each one a panel per method in
`--methods`. Every word is shaded by that method's exact additive contribution
to its own decision -- the decomposition Stage 2 scores as word AUPRC
(`results/dla_auprc.csv`), aggregated onto words with the pinned
`audit.word_score_aggregation`. Nothing post-hoc, no attribution method, no
SAE: it is the method's own readout term by term, and the terms sum to the
score it predicts. Where a psychiatrist annotated an evidence span, that span
is underlined in green on the same words, and the left column names whose
readout each panel is.

THE PAIR IS THE POINT. The default page draws both sides of the study's
central tie -- Full FT and the frozen attention-pool logistic match on AUROC
(0.927 vs 0.932 on Gemma-2-2B) -- on the same two posts, in the same layout,
under the same colour rule. The two panels of a case differ by the weights
under the words and by nothing else:

  * `full_ft` reloads the checkpoint Stage 2 scored and reads its head through
    `src.utils.dla.token_contributions`.
  * `attention_probe` (Attn-Linear) is refitted out of fold on the cached
    activations exactly as `src.stage2.dla_auprc` refits it, and its terms are
    `alpha_t * (w . X_t)`, which sum to its score by the identity
    `src.probe.attention_probe.verify_decomposition` enforces per fold.

    A probe keeps no checkpoint, so this is a fit of the same procedure rather
    than a reload of the published cell. It is pinned to the CPU, as Stage 1
    and Stage 2 pin it, which is what makes it the same fit: fitted on a GPU
    the same seed lands a few hundredths away. `check_refit` still requires the
    refit score to fall on the published fold threshold's own side of the line
    before any panel is captioned with a decision. `plan_devices` refuses a
    page that would have to load a backbone AND fit a probe in one process,
    because only one of the two device pins can win; drawing the methods one
    at a time costs nothing, since the cache makes the second pass free.

On the two posts drawn, both methods make the SAME two decisions -- a true
positive and a false positive -- and rank the psychiatrist span very
differently, which is the study's claim in one page: the tie in AUROC is not
a tie in evidence.

The cases run down Full FT's own decisions. A case heading states what the
psychiatrist said -- the one fact no method's weights can change -- and each
panel's left-column label is DERIVED from that label and from that method's
own published decision, never from the pool an id came from, so two methods
that disagree on a post get two different labels over the same words. The
"wrong evidence" flag is derived too, and written only where that method
ranks the span below the chance a random ranker gets on that post, so a label
cannot contradict the numbers beside it. Under Full FT the cases read:

  ROW 1  A TRUE POSITIVE WITH DISTORTED EVIDENCE. The clinician marked
         loneliness present and annotated the span; the arm agrees on the
         label and ranks that span near the BOTTOM of its own words, so the
         post's word AUPRC lands below the chance a random ranker gets on it
         (`word_gold_base_rate`). The right decision, none of the evidence.

  ROW 2  A FALSE POSITIVE. The clinician marked loneliness absent and
         annotated nothing, and the arm predicts `present` anyway. With no
         span to underline, the row is read off the shading alone: the words
         carrying the decision are generic distress and function words.

The two rows fail in different ways on the same axis: row 1 is right on the
label and wrong on the reasons, row 2 manufactures evidence out of words no
psychiatrist marked. Row 1 is the uncomfortable one -- the arm is RIGHT
there, and its reasons are no better.

`plot/plot_error_plausibility.py` is the population-level version of the same
claim: on word AUPRC against the psychiatrist's span, Full FT sits far below
every mean-pooled readout that reads a mid-depth site -- including the arm that
is the same fine-tuning recipe with its head moved to that site.

CHOOSING THE POSTS. `--list-candidates` prints the two ranked pools the rows
are drawn from, straight off the published cells, so the IDs in the figure are
a selection from a printed list rather than an unexplained pair:

    uv run python plot/plot_feature_distortion.py --list-candidates

  * false positives come from those decisions filtered to `label == absent`
    and `post_pred == present`, ranked by logit, i.e. by how confident the
    error was. They cannot be ranked by attribution quality the way the other
    pool is: an `absent` post carries no annotated span, so it has no gold
    word, and AUPRC is undefined on a ranking with no positive class.
  * distorted true positives come from
    `results/stage2/plaus_ft_{model}_fold*.parquet` -- the per-post cells the
    published word AUPRC is the mean of -- filtered to posts the arm scores
    positive and ranked by word AUPRC against their own chance base rate.

Both pools are restricted to posts short enough to render whole, because
a truncated post would show the reader a decision made partly on words that
are not on the page. `--tp-id` / `--fp-id` override the defaults.
The pools are read off fine-tuning cells and exist for the arms only; the
probe figure draws the arm's posts on purpose, so that the two figures differ
by the method and by nothing else.

CACHING. The word contributions are computed once per (method, model, post)
and cached under cache/feature_distortion/; re-running to restyle either
figure needs no GPU, no --allow-model-process and, for the probe, no refit.
`--recompute` forces a fresh forward pass or a fresh fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for style.py

from src.utils.common import (  # noqa: E402
    ARM_OF,
    PROBES,
    artifact_path,
    load_config,
    method_label,
    model_label,
    pin_visible_devices,
    require_model_process_approval,
    resolve_model,
)
from src.utils.data_prep import load_corpus, words  # noqa: E402

# Rendered figures, not code: plot/ holds the scripts, plot/figs/ the PNGs.
OUT = Path(__file__).resolve().parent / "figs"
CACHE = ROOT / "cache" / "feature_distortion"

# The method the figure is about by default. Full FT is the study's worst
# plausibility cell (results/dla_auprc.csv) and its best-known recipe, which is
# the whole point: the distortion is not a broken run, it is what the standard
# recipe does. It is drawn against the frozen attention-pool logistic, which
# is the comparison the study turns on.
# Both readouts of the study's central tie: Full FT and the frozen
# attention-pool logistic match on AUROC (0.927 vs 0.932 on Gemma-2-2B), so
# every post is drawn twice, once under each, and the page is about what
# separates them when the number does not.
DEFAULT_METHODS = ("full_ft", "attention_probe")

# The defaults, taken from the top of the two `--list-candidates` pools.
#
# lx2333 (fold 6): "My closest friends have been distant lately." Annotated
# span `friends have been distant`. Full FT predicts present at logit +18.5
# and still ranks that span below chance -- 28th, 42nd, 44th and 45th of 46.
#
# lx3038 (fold 8): the arm's most confident false positive at a renderable
# length -- logit +17.5 on a post the clinician marked absent. Its five
# strongest words are `down!`, `am`, `What`, `nothing` and `die!`: generic
# distress and function words, on a post whose only relational content is a
# family complaint and a YouTube channel.

# The same two posts under `attention_probe`, the panel beneath each, which
# is why they are worth drawing twice: the frozen probe makes the same two
# decisions and ranks the psychiatrist's span 3.5x chance on lx2333 where
# Full FT ranks it at 0.7x -- `distant` is the probe's single largest term
# (+4.13) and the annotated span's word AUPRC is 0.302 against 0.087. lx3038
# -- the same false alarm on the same generic distress words (`down!`,
# `interest`, `die`), which is the honest half of the pair: the probe's
# evidence is better, and it is not a detector that stopped making mistakes.

DEFAULT_TP_ID = "lx2333"
DEFAULT_FP_ID = "lx3038"

# Rendering budget. A post past this many words does not fit two panels at a
# readable size, and a figure that clips a post is showing a decision made
# partly off-page.
MAX_WORDS = 80

# Same ink/paper as plot/style.py. The diverging pair is this figure's own --
# blue pushes the logit away from "lonely", red pushes it toward -- and the
# annotation green is used for nothing else, so the clinician's ink can never
# be confused with the model's.
INK = "#1C1917"
PAPER = "#FFFFFF"
NEG_BLUE = "#4E79A7"
POS_RED = "#E15759"
NEUTRAL = "#FBFAF9"
GOLD_GREEN = "#59A14F"
# The same green at low opacity, pre-mixed on white: a band drawn behind text
# has to be light enough to read through, and alpha on a patch would darken
# wherever two rectangles happened to overlap.
GOLD_BAND = "#CFE3CC"
ARIAL_DIR = Path("/usr/share/fonts/truetype/liberation2")

DIVERGING = LinearSegmentedColormap.from_list(
    "contribution", [NEG_BLUE, NEUTRAL, POS_RED])


def _style() -> None:
    for fname in (
        "LiberationSans-Regular.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-BoldItalic.ttf",
    ):
        path = ARIAL_DIR / fname
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 12,
        }
    )


# ===================================================== the arm's own evidence


# This figure's own reader-facing name for the frozen probe. `plot/style.py`'s
# shared DISPLAY dict calls it "Attn-Linear" because the pooling-comparison
# figures need the "Attn-" qualifier to set it apart from the mean-pooled
# `linear_probe`; this page draws only one probe, so the qualifier is noise a
# reader has to parse instead of information. Overridden here rather than in
# the shared dict, which every other figure's axis ticks also read.
FIGURE_DISPLAY = {"attention_probe": "Linear Probe"}


def display_name(method: str) -> str:
    """The name the figure calls a method, from the study's one naming map.

    `plot/style.py` holds it, and the figures' axis ticks read the same dict:
    a method renamed there cannot end up named two ways across the figures.
    The newlines in it are tick-label line breaks, which running prose has no
    use for.
    """
    if method in FIGURE_DISPLAY:
        return FIGURE_DISPLAY[method]
    from style import DISPLAY  # noqa: E402  (plot/ is a directory, not a package)

    return DISPLAY.get(method, method).replace("\n", " ")


def _cache_path(method: str, model: str, post_id: str, fold: int,
                reduction: str) -> Path:
    return CACHE / (f"{method_label(method)}_{model_label(model)}_"
                    f"fold{fold}_{post_id}_{reduction}.json")


def word_contributions(post, method: str, model: str, fold: int,
                       reduction: str, *, recompute: bool,
                       allow_model_process: bool) -> dict:
    """One method's own additive term for every word of one post.

    Two paths, because the study's two families keep their weights in
    different places, and both end in the same record: a fine-tuning arm
    reloads the checkpoint Stage 2 scored and reads its head term by term; a
    frozen probe is refitted out of fold on the cached activations, exactly as
    `src.stage2.dla_auprc` refits it. Neither is an attribution method — both
    are the decomposition the arm's own decision is a sum of.
    """
    cached = _cache_path(method, model, post["id"], fold, reduction)
    if cached.exists() and not recompute:
        return json.loads(cached.read_text())
    record = (_arm_contributions(post, method, model, fold, reduction,
                                 allow_model_process=allow_model_process)
              if method in ARM_OF else
              _probe_contributions(post, method, model, fold, reduction))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(record, indent=2))
    print(f"cached {cached.relative_to(ROOT)}  (score {record['logit']:+.2f}, "
          f"identity residual {record['decomposition_max_abs_error']:.2e})")
    return record


def _arm_contributions(post, method: str, model: str, fold: int,
                       reduction: str, *, allow_model_process: bool) -> dict:
    """A fine-tuning arm's terms, read off the checkpoint Stage 2 scored.

    The computation is `src.utils.dla`'s, not a second copy of it: the arm's
    checkpoint is reloaded, its own `token_states` and head give the exact
    per-token terms, and `aggregate_word_scores` folds them onto words under
    the pinned reduction.
    """
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.common import checkpoint_path
    from src.utils.dla import decomposition_errors, token_contributions
    from src.utils.sae_features import (aggregate_word_scores,
                                        align_post_tokens)

    cfg = load_config()
    require_model_process_approval(
        cfg, cli_override=allow_model_process,
        command="plot/plot_feature_distortion.py")

    arm = ARM_OF[method]
    checkpoint = checkpoint_path(arm, model, fold)
    if not checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint} — the figure reads the SAME "
            f"weights Stage 2 scored. Train it with `python -m src.stage1.train "
            f"--method {method} --cells {fold} --keep-checkpoint`.")

    text = str(post["text"])
    max_len = int(cfg["full_finetune"]["max_len"])
    classifier, tok = ft.load_checkpoint(checkpoint, model)
    try:
        device = next(classifier.parameters()).device
        ids, attn, post_mask = ft.build_inputs(tok, [text], max_len)
        ids, attn, post_mask = ids.to(device), attn.to(device), post_mask.to(device)
        with torch.no_grad():
            states = classifier.token_states(ids, attn, post_mask)
            logits = classifier(ids, attn, post_mask)
            pooling_weights = (
                None if getattr(classifier, "pooling", "mean") == "mean"
                else classifier.pooling_weights(states, post_mask))
            contributions = token_contributions(
                states, post_mask, classifier.head.weight,
                classifier.head.bias, weights=pooling_weights)
            # The identity is the reason these numbers are the arm's evidence
            # and not an attribution of it: the terms drawn below plus the
            # head bias reproduce the logit the arm actually predicted.
            residual = float(decomposition_errors(
                contributions, post_mask, classifier.head.bias, logits)[0])
            scores = contributions[0].float().cpu().numpy()
            logit = float(logits.float().cpu().numpy().reshape(-1)[0])
    finally:
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    alignment = align_post_tokens(
        tok, text, max_len=max_len,
        span_start=int(post["span_start"]), span_end=int(post["span_end"]))
    word_scores = aggregate_word_scores(scores, alignment, reduction=reduction)
    return _record(post, method, ARM_OF[method], model, fold, reduction,
                   logit=logit, bias=float(classifier_bias(checkpoint)),
                   residual=residual, truncated=bool(alignment.truncated),
                   word_scores=word_scores)


# One context per (model, layer) per process: `build_context` reads every
# cached token vector of the corpus, and the two posts of a figure sit in
# two different folds off the same activations.
_PROBE_CONTEXT = {}


def _probe_context(model: str, seed: int = 0):
    from src import probe as probing
    from src.utils.common import DENSE_PROBE_LAYER

    layer = DENSE_PROBE_LAYER[model]
    if (model, layer, seed) not in _PROBE_CONTEXT:
        print(f"loading {model} layer {layer} activations", flush=True)
        _PROBE_CONTEXT[(model, layer, seed)] = probing.build_context(
            model, layer, load_config(), seed)
    return _PROBE_CONTEXT[(model, layer, seed)]


def _probe_contributions(post, method: str, model: str, fold: int,
                         reduction: str, seed: int = 0) -> dict:
    """A frozen probe's terms, refitted out of fold on cached activations.

    A probe keeps no checkpoint -- Stage 2 refits it out of fold every time it
    scores it -- so this row is drawn from a fit of the same fitter, the same
    seed and the same held-out fold, on the CPU, with no backbone in the
    process. That is one fit of the procedure the published mean is over, not
    a reload of the exact cell, and `main` checks the refit lands on the same
    side of the published fold threshold before drawing a decision from it.

    The bias is measured, not assumed: attention pooling's identity is
    `sum_t alpha_t (w . X_t) + b`, so scoring one reference matrix both ways
    and subtracting recovers `b` from the two callables independently, and the
    residual reported here is then a real check on THIS post rather than an
    arithmetic tautology.
    """
    from src import probe as probing
    from src.utils.sae_features import aggregate_word_scores

    ctx = _probe_context(model, seed)
    fitted = probing.FITTERS[method](probing.for_fold(ctx, fold))
    if fitted.token_scores is None:
        raise SystemExit(
            f"{method} has no additive token decomposition "
            f"({fitted.token_source}) — this figure draws a method's own "
            f"terms, and there are none to draw.")

    post_id = post["id"]
    index = ctx.index_of[post_id]
    scores = np.asarray(fitted.token_scores(index), dtype=np.float64)
    reference = ctx.acts[ctx.all_ids[0 if ctx.all_ids[0] != post_id else 1]]
    bias = float(fitted.score_vectors(reference["vecs"])
                 - np.sum(fitted.token_vectors_scores(reference["vecs"])))
    logit = float(fitted.post_scores(np.array([index]))[0])
    targets = probing.word_targets(
        ctx.acts[post_id]["offsets"], str(post["text"]),
        int(post["span_start"]), int(post["span_end"]))
    word_scores = aggregate_word_scores(scores, targets.alignment,
                                        reduction=reduction)
    return _record(post, method, method, model, fold, reduction,
                   logit=logit, bias=bias,
                   residual=abs(logit - (float(scores.sum()) + bias)),
                   truncated=False, word_scores=word_scores)


def _record(post, method, arm, model, fold, reduction, *, logit, bias,
            residual, truncated, word_scores) -> dict:
    return {
        "id": post["id"],
        "method": method,
        "arm": arm,
        "model": model,
        "fold": fold,
        "word_aggregation": reduction,
        "logit": logit,
        "head_bias": bias,
        "decomposition_max_abs_error": residual,
        "truncated": truncated,
        # NaN for a word no token reached (punctuation, or past truncation);
        # JSON has no NaN, so those come back as None and are drawn unscored.
        "word_scores": [None if not np.isfinite(v) else float(v)
                        for v in word_scores],
    }


def classifier_bias(checkpoint: Path) -> float:
    """The head intercept, read off the saved head rather than the live model.

    It is not distributed over tokens (`token_contributions` refuses to), so it
    is carried alongside the terms: a reader checking that the shaded words sum
    to the logit needs the constant they sum to it *with*.
    """
    import torch

    head = torch.load(checkpoint / "head.pt", map_location="cpu")
    return float(head["head.bias"].reshape(-1)[0])


# ============================================================ post selection


def _cells(pattern: str):
    import glob as globlib

    import pandas as pd

    paths = sorted(globlib.glob(pattern))
    if not paths:
        raise SystemExit(f"no cells matching {pattern}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _scored_words(text: str) -> int:
    return len([w for w in words(text) if w["norm"]])


def _arm_only(method: str) -> None:
    """The candidate pools are read off arm cells; say so rather than crash.

    A frozen probe publishes no per-post `plaus_*` cell and no per-fold
    `detect_*` cell of the shape these pools read, and it is not what the
    listing is for anyway: the probe figure draws the arm's posts on purpose,
    so that the two figures differ by the method and by nothing else.
    """
    if method not in ARM_OF:
        raise SystemExit(
            f"--list-candidates reads the fine-tuning cells, and {method} is "
            f"a frozen probe with none. List the arm's pools "
            f"(--methods full_ft --list-candidates); the probe's panel is "
            f"drawn on whatever posts that listing chooses.")


def _decision_pool(lonxplain, method: str, model: str, label: str, pred: str,
                   ascending: bool):
    """Posts where the clinician said `label` and the arm predicted `pred`.

    Ranked by logit -- by how confident the decision was -- and not by
    attribution quality, because for the two `absent` pools there is none to
    rank by: an `absent` post carries no annotated span, so every word is a
    non-gold word and a ranking with no positive class has no average
    precision.
    """
    cells = _cells(str(artifact_path("detect", ARM_OF[method], model,
                                     fold="*")))
    cells = cells[cells["corpus"] == "lonxplain"]
    pool = cells[(cells["label"] == label)
                 & (cells["post_pred"] == pred)].copy()
    indexed = lonxplain.set_index("id")
    pool["n_words"] = pool["id"].map(indexed["text"].map(_scored_words))
    pool = pool[pool["n_words"] <= MAX_WORDS]
    return pool.sort_values("post_score", ascending=ascending)


def false_positive_pool(lonxplain, method: str, model: str):
    """False alarms, most confident (most positive logit) first."""
    return _decision_pool(lonxplain, method, model, "absent", "present",
                          ascending=False)


def distorted_true_positive_pool(method: str, model: str):
    """Posts the arm calls present while ranking the annotated span at chance.

    `word_auprc` is the arm's own ranking of its own words against the
    psychiatrist span; `word_gold_base_rate` is what a random ranker scores on
    that same post. Ranking by their ratio, rather than by the raw AUPRC, keeps
    a long post with three gold words from outranking a short one purely
    because chance is lower on it.
    """
    cells = _cells(str(artifact_path("plaus", ARM_OF[method], model,
                                     fold="*")))
    pool = cells[(cells["clean_logit"] > 0)
                 & (~cells["truncated"])
                 & (cells["n_gold_words"] >= 2)
                 & (cells["n_words_scored"] <= MAX_WORDS)].copy()
    pool["auprc_over_chance"] = pool["word_auprc"] / pool["word_gold_base_rate"]
    return pool.sort_values("auprc_over_chance")


def list_candidates(lonxplain, method: str, model: str, reduction: str, *,
                    score: bool, allow_model_process: bool) -> None:
    """Print the two pools, each ranked by the criterion its row is chosen on.

    They are not ranked the same way. A distorted true positive has an
    annotated span, so its rows sort by how badly the arm ranks that span
    against its own chance base rate. The `absent` pool has none -- the
    clinician wrote nothing down -- so there is no gold word, no AUPRC, and
    the only ordering available is how confident the decision was.

    Showing a candidate's strongest words needs its contributions, so that
    line appears only when the cache entry already exists.
    `--score-candidates` computes the missing ones (one forward pass each,
    behind the study gate).
    """
    lookup = lonxplain.set_index("id")

    def described(post_id):
        source = lookup.loc[post_id]
        return {"id": post_id, **{k: source[k] for k in
                                  ("text", "label", "span_text",
                                   "span_start", "span_end", "fold")}}

    def scored_record(post):
        """The post's contributions, or None when nothing has computed them."""
        fold = int(post["fold"])
        if not (score
                or _cache_path(method, model, post["id"], fold,
                               reduction).exists()):
            return None
        return word_contributions(post, method, model, fold, reduction,
                                  recompute=False,
                                  allow_model_process=allow_model_process)

    _arm_only(method)
    fp = false_positive_pool(lonxplain, method, model)
    print(f"\nFALSE POSITIVES — clinician `absent`, {method} predicts "
          f"`present` ({len(fp)} at <= {MAX_WORDS} words), most confident "
          f"error first\n")
    for _, row in fp.head(12).iterrows():
        post = described(row["id"])
        print(f"  {post['id']}  fold {post['fold']}  "
              f"logit {row['post_score']:+7.2f}  {row['n_words']:3d} words")
        record = scored_record(post)
        if record is not None:
            print(f"      the arm's strongest words: "
                  f"{top_words(post, record)}")
        print(f"      {str(post['text']).replace(chr(10), ' ')[:150]}...")

    tp = distorted_true_positive_pool(method, model)
    print(f"\nDISTORTED TRUE POSITIVES — clinician `present`, {method} predicts "
          f"`present`, span ranked at or below chance ({len(tp)} at "
          f"<= {MAX_WORDS} words)\n")
    for _, row in tp.head(12).iterrows():
        post = lookup.loc[row["id"]]
        print(f"  {row['id']}  fold {row['fold']}  logit {row['clean_logit']:+7.2f}"
              f"  {row['n_words_scored']:3d} words  word AUPRC "
              f"{row['word_auprc']:.3f} vs chance "
              f"{row['word_gold_base_rate']:.3f} "
              f"({row['auprc_over_chance']:.2f}x)")
        print(f"      span: {post['span_text']!r}")
        print(f"      {str(post['text']).replace(chr(10), ' ')[:150]}...")
    print()


def decisions(method: str, model: str):
    """Every held-out decision this method published, indexed by post.

    The two families file their cells differently -- an arm writes one parquet
    per fold with `post_score`/`post_pred` and no threshold, because its
    decision is the sign of its logit; a probe writes one file per model
    carrying the `threshold` `src.stage1.train` fitted on the training folds,
    because its decision is not. Both are renamed onto the same three columns
    here so a row can be drawn without knowing which family it came from, and
    the threshold column is what tells the note which sentence to write.
    """
    import pandas as pd

    if method in ARM_OF:
        cells = _cells(str(artifact_path("detect", ARM_OF[method], model,
                                         fold="*")))
        cells = cells[cells["corpus"] == "lonxplain"].copy()
        cells["threshold"] = 0.0
        return cells.set_index("id")
    path = artifact_path("detect", method, model)
    if not path.exists():
        raise SystemExit(
            f"no decision cells at {path} — the figure names the decision "
            f"the study published. Write them with `python -m "
            f"src.stage1.detection --method {method}`.")
    cells = pd.read_parquet(path).rename(
        columns={"score": "post_score", "pred": "post_pred"})
    return cells.set_index("id")


_DECISIONS = {}


def decision_for(method: str, model: str, post_id: str):
    """This method's published decision on one post, or None if it has none."""
    if (method, model) not in _DECISIONS:
        _DECISIONS[(method, model)] = decisions(method, model)
    table = _DECISIONS[(method, model)]
    return table.loc[post_id] if post_id in table.index else None


def plan_devices(methods, posts, model: str, reduction: str, *,
                 recompute: bool) -> None:
    """Pin the CPU when nothing on the page needs a GPU, and say so when both do.

    Refitting a probe is a CPU job in every Stage 1/2 CLI (`--gpu` is opt-in
    there), and being the same fit is the whole claim: on a GPU the same seed
    lands a few hundredths away from the published cell. Reloading an arm's
    checkpoint, on the other hand, needs the card. The two cannot be pinned at
    once, so a page that would have to compute both is refused rather than
    quietly fitting the probe on the GPU -- and refusing costs nothing, since
    the first pass caches what the second one reads.

    Must run before anything imports torch, which is why it takes the posts
    rather than the records: it decides from the cache, not from a load.
    """
    missing = [(method, post["id"]) for method in methods for post in posts
               if recompute or not _cache_path(
                   method, model, post["id"], int(post["fold"]),
                   reduction).exists()]
    arms = [m for m, _ in missing if m in ARM_OF]
    probes = [m for m, _ in missing if m not in ARM_OF]
    if arms and probes:
        raise SystemExit(
            f"this page would have to run {sorted(set(arms))} on a GPU and "
            f"fit {sorted(set(probes))} on the CPU in one process. Draw them "
            f"one at a time (--methods {' '.join(sorted(set(probes)))}, then "
            f"--methods {' '.join(sorted(set(arms)))}); the cache makes the "
            f"second pass free.")
    if not arms:
        pin_visible_devices("cpu", need_gpu=False)


def check_refit(post_id: str, record: dict, decision) -> None:
    """A refitted probe has to reach the decision the study published.

    A fine-tuning arm reloads the weights Stage 2 scored, so its logit is that
    cell's logit. A probe is refitted here (see `_probe_contributions`), and
    two fits of one procedure agree closely but not to the last digit -- the
    corpus was re-encoded once already, and `results/dla_auprc.csv`'s probe
    rows moved with it. Close enough to draw is therefore not an assumption:
    the refit score has to fall on the same side of the published fold
    threshold as the published score, or the row would be captioned with a
    decision the drawn terms do not make.
    """
    published, threshold = float(decision["post_score"]), float(
        decision["threshold"])
    refit = float(record["logit"])
    if (refit > threshold) != (published > threshold):
        raise SystemExit(
            f"{post_id}: refitted score {refit:+.3f} and published score "
            f"{published:+.3f} straddle this fold's threshold "
            f"{threshold:+.3f} — the refit makes a different decision than "
            f"the study published, and the row cannot be drawn from both.")


def top_word_indices(record, n: int = 5) -> list[int]:
    """Word positions of the arm's `n` largest terms, largest first."""
    values = np.array(
        [np.nan if v is None else float(v) for v in record["word_scores"]],
        dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, -np.inf))[::-1]
    return [int(i) for i in order[:n] if np.isfinite(values[i])]


def top_words(post, record, n: int = 5) -> str:
    """The arm's `n` strongest words on this post, strongest first."""
    post_words = words(str(post["text"]))
    values = np.array(
        [np.nan if v is None else float(v) for v in record["word_scores"]],
        dtype=float)
    return ", ".join(f"{post_words[i]['text']} ({values[i]:+.2f})"
                     for i in top_word_indices(record, n))


# ================================================================== rendering
#
# Matplotlib has no flowed text, so the post is laid out one word at a time and
# measured with the real renderer. That makes the number of LINES a panel takes
# something the figure can only learn by drawing, which is why the layout runs
# in two passes: `measure_lines` wraps the post on a throwaway canvas of the
# final panel width, and the figure is then built at exactly the height its two
# posts need. Sizing by a word-count guess instead is what leaves a text figure
# either clipped or two-thirds white space.

MARGIN_L, MARGIN_R = 0.10, 0.10   # inches; everything below is inches too
# The left column that names whose readout a panel is. Every panel starts to
# the right of it, so the eye reads down one edge and the method names line
# up across the whole page.
GUTTER = 1.55
TEXT_MEASURE = 10.6      # the width the post is wrapped to
# The figure is exactly as wide as its text: margins, the name column and the
# measure. Fixing the width first and wrapping inside it is what left the
# earlier page a quarter white space on the right. There is no legend: the
# shading's sense (red toward "lonely", blue away) and the green underline
# for the annotated span are stated in the caption.
FIG_WIDTH = MARGIN_L + GUTTER + TEXT_MEASURE + MARGIN_R
TOP_MARGIN = 0.05         # figure edge to the first case heading
CASE_TITLE_H = 0.30       # the post heading over a case's panels
METHOD_GAP = 0.02         # between two methods reading the same post
CASE_GAP = 0.06           # between two posts (METHOD_GAP already trails)
BOTTOM_MARGIN = 0.05      # last panel to the figure edge
LINE_SPACING = 1.62       # baseline-to-baseline, in multiples of the font size


def _panel_width() -> float:
    """The measure the post is wrapped to.

    Running text set wider than this would run past the length an eye tracks
    back from comfortably; the rows are read as prose, not scanned as a table.
    The figure width is derived from it, so the two cannot disagree.
    """
    return TEXT_MEASURE


def _line_height(fontsize: float) -> float:
    return fontsize * LINE_SPACING / 72.0


def _layout(ax, fig, tokens, fontsize):
    """Word positions and widths in axes fractions, plus the line count.

    Pure measurement: a word is drawn, its width read off the real renderer,
    and the pen wraps when the next word would cross the right edge. Nothing is
    styled here, so the same arithmetic serves the measuring pass and the two
    panels that share a post.
    """
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def axes_width(artist) -> float:
        box = artist.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform([(box.x0, box.y0), (box.x1, box.y1)])
        return x1 - x0

    probe = ax.text(0, -5, "n n", fontsize=fontsize, transform=ax.transAxes)
    two_words = axes_width(probe)
    probe.set_text("nn")
    space = two_words - axes_width(probe)
    probe.remove()

    placed, pen_x, line = [], 0.0, 0
    for token in tokens:
        artist = ax.text(0, -5, token, fontsize=fontsize,
                         transform=ax.transAxes)
        width = axes_width(artist)
        artist.remove()
        if pen_x > 0 and pen_x + width > 1.0:
            pen_x, line = 0.0, line + 1
        placed.append((pen_x, line, width))
        pen_x += width + space
    return placed, line + 1, space


def measure_lines(tokens, fontsize) -> int:
    """How many lines `tokens` wrap to in a panel of the final width."""
    fig = plt.figure(figsize=(_panel_width(), 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.canvas.draw()
    _, lines, _ = _layout(ax, fig, tokens, fontsize)
    plt.close(fig)
    return lines


def _axes_height_inches(ax, fig) -> float:
    return ax.get_position().height * fig.get_figheight()


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_panel(ax, fig, tokens, *, fontsize, line_step, fills, band,
               band_fill):
    """Draw the post as shaded words, with the annotated span marked.

    The shading is drawn as explicit rectangles from the measured word
    positions rather than as matplotlib's per-text `bbox`. A bbox carries its
    own padding, so adjacent boxes overlap: same-coloured neighbours merge into
    one blob and swallow the space between them, and a later box paints over
    the tail of the word before it. Rectangles placed at the measured extents
    do neither, and they let the psychiatrist's span be drawn as ONE continuous
    mark per line instead of a row of separate word tiles.

    `band_fill` is how that span is marked, and it differs by panel for a
    reason. In the annotation panel the words carry no other colour, so the
    span is a filled green band. In the model panel every word is already
    filled with the arm's own contribution -- a second fill would either hide
    that colour or be hidden by it -- so the span is underlined instead, and
    the arm's shading stays readable straight through the clinician's mark.
    """
    _blank(ax)
    fig.canvas.draw()
    placed, _, space = _layout(ax, fig, tokens, fontsize)
    # A word's rectangle in axes fractions. Height and width are measured in
    # DIFFERENT units here -- the axes is not square, so one fraction of its
    # height is not one fraction of its width -- and each side therefore takes
    # its own: the glyph box is one font size tall, and the horizontal padding
    # is a fraction of the inter-word space. Padding x by a vertical quantity
    # is what put the ends of the psychiatrist's band in the middle of a word.
    glyph = (fontsize / 72.0) / _axes_height_inches(ax, fig)
    pad = 0.30 * space

    def rect(x, width, line, **kwargs):
        y = 1.0 - line * line_step
        return Rectangle((x - pad, y - 1.06 * glyph), width + 2 * pad,
                         1.34 * glyph, transform=ax.transAxes, **kwargs)

    for token, (pen_x, line, width), fill in zip(tokens, placed, fills):
        if fill is not None:
            ax.add_patch(rect(pen_x, width, line, facecolor=fill,
                              edgecolor="none", zorder=1))
        ax.text(pen_x, 1.0 - line * line_step, token, fontsize=fontsize,
                transform=ax.transAxes, va="top", ha="left", color=INK,
                zorder=4)

    # Contiguous same-line runs are one mark, so the span reads as a marked
    # passage rather than as a row of marked words.
    for run in _runs(band, placed):
        x0, line, _ = placed[run[0]]
        x1, _, width = placed[run[-1]]
        y = 1.0 - line * line_step
        if band_fill:
            ax.add_patch(rect(x0, x1 + width - x0, line, facecolor=GOLD_BAND,
                              edgecolor="none", zorder=1))
        else:
            ax.plot([x0 - pad, x1 + width + pad], [y - 1.10 * glyph] * 2,
                    color=GOLD_GREEN, linewidth=2.4, solid_capstyle="butt",
                    transform=ax.transAxes, clip_on=False, zorder=5)


def _runs(mask, placed) -> list[list[int]]:
    """Maximal runs of `mask` that are adjacent AND on the same drawn line."""
    out: list[list[int]] = []
    for i, flagged in enumerate(mask):
        if not flagged:
            continue
        if (out and i == out[-1][-1] + 1
                and placed[i][1] == placed[out[-1][-1]][1]):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def shading(post, record):
    """Per-word fill colours and the psychiatrist's span mask, for one post."""
    post_words = words(str(post["text"]))
    span_start, span_end = int(post["span_start"]), int(post["span_end"])
    has_span = span_start >= 0 and span_end > span_start
    gold = [bool(has_span and w["start"] < span_end and w["end"] > span_start)
            for w in post_words]

    values = np.array(
        [np.nan if s is None else float(s) for s in record["word_scores"]],
        dtype=float)
    finite = np.isfinite(values)
    # Symmetric about zero, so the midpoint of the colormap IS "contributes
    # nothing", and scaled by the post's own largest term: the shading compares
    # words within one decision and is never a scale across posts.
    scale = float(np.max(np.abs(values[finite]))) if finite.any() else 1.0
    norm = Normalize(vmin=-scale, vmax=scale)
    # None, not white: a word no token reached (punctuation, or past
    # truncation) carries no contribution to shade, and drawing it white would
    # claim the arm scored it at zero.
    fills = [DIVERGING(norm(v)) if np.isfinite(v) else None for v in values]
    return [w["text"] for w in post_words], fills, gold, values, finite


CONFUSION = {
    ("present", "present"): "true positive",
    ("present", "absent"): "false negative",
    ("absent", "present"): "false positive",
    ("absent", "absent"): "true negative",
}


def prediction(record, decision) -> str:
    """The decision this method made on this post, as the study published it."""
    if decision is not None:
        return str(decision["post_pred"])
    return "present" if record["logit"] > 0 else "absent"


def case_heading(post) -> str:
    """What the psychiatrist said about this post, above every method's row.

    The clinician's reading is the one fact the methods are compared against,
    so it is stated once per post rather than once per panel -- and it is the
    only line in a case block that no method's weights can change.
    """
    return "Lonely present post" if str(post["label"]) == "present" else "Lonely absent post"




def main() -> int:
    ap = argparse.ArgumentParser(
        description="The words Full FT actually decides on "
                    "-> plot/figs/fig_feature_distortion.png")
    ap.add_argument("--model", default="gemma2")
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                    choices=tuple(ARM_OF) + tuple(PROBES),
                    help="whose decompositions to draw, one panel each under "
                         f"every post (default: {' '.join(DEFAULT_METHODS)}); "
                         "a frozen probe is refitted out of fold on cached "
                         "activations, an arm is reloaded from its checkpoint")
    ap.add_argument("--tp-id", default=DEFAULT_TP_ID,
                    help="true positive whose annotated span the arm ranks at "
                         "chance")
    ap.add_argument("--fp-id", default=DEFAULT_FP_ID,
                    help="false positive: annotated `absent`, arm says present")
    ap.add_argument("--word-aggregation", default=None,
                    choices=("max", "mean", "sum"),
                    help="sub-word -> word reduction (default: the "
                         "audit.word_score_aggregation pin)")
    ap.add_argument("--list-candidates", action="store_true",
                    help="print both selection pools and exit")
    ap.add_argument("--score-candidates", action="store_true",
                    help="--list-candidates: also show the arm's strongest "
                         "words for candidates with no cached contributions "
                         "yet (one forward pass each; needs "
                         "--allow-model-process)")
    ap.add_argument("--recompute", action="store_true",
                    help="re-run the forward pass instead of reading the cache")
    ap.add_argument("--fontsize", type=float, default=12.5)
    ap.add_argument("--out", default="fig_feature_distortion.png",
                    help="file name under plot/figs")
    ap.add_argument("--allow-model-process", action="store_true")
    args = ap.parse_args()

    model = resolve_model(args.model)
    methods = list(dict.fromkeys(args.methods))   # de-duplicated, order kept
    cfg = load_config()
    reduction = (args.word_aggregation
                 or cfg["audit"]["word_score_aggregation"])
    lonxplain, _ = load_corpus("lonxplain")

    if args.list_candidates:
        list_candidates(lonxplain, methods[0], model, reduction,
                        score=args.score_candidates,
                        allow_model_process=args.allow_model_process)
        return 0

    indexed = lonxplain.set_index("id")
    posts = []
    # Top to bottom: the evidence it had and did not use, the evidence it
    # invented.
    for post_id in (args.tp_id, args.fp_id):
        if post_id not in indexed.index:
            raise SystemExit(f"{post_id} is not a LonXplain post id")
        source = indexed.loc[post_id]
        posts.append({"id": post_id, **{k: source[k] for k in
                                        ("text", "label", "span_text",
                                         "span_start", "span_end", "fold")}})
    plan_devices(methods, posts, model, reduction, recompute=args.recompute)

    cases = []
    for post in posts:
        panels = []
        for method in methods:
            record = word_contributions(
                post, method, model, int(post["fold"]), reduction,
                recompute=args.recompute,
                allow_model_process=args.allow_model_process)
            decision = decision_for(method, model, post["id"])
            if decision is not None and method not in ARM_OF:
                check_refit(post["id"], record, decision)
            panels.append((method, record, decision))
        cases.append((post, panels))

    _style()
    line_h = _line_height(args.fontsize)
    panel_w = _panel_width()

    drawn = []
    for post, panels in cases:
        rows = []
        for method, record, decision in panels:
            tokens, fills, gold, values, finite = shading(post, record)
            rows.append(dict(
                method=method, record=record, tokens=tokens, fills=fills,
                gold=gold, pred=prediction(record, decision),
                lines=measure_lines(tokens, args.fontsize)))
        # Both methods make the same decision on every post this figure
        # draws, so the first row's decision speaks for the case.
        confusion = CONFUSION[(str(post["label"]), rows[0]["pred"])]
        drawn.append(dict(post=post, heading=case_heading(post), rows=rows,
                          confusion=confusion))

    # Every panel's own height, measured before the figure exists: a row is as
    # tall as its gutter label wrapped plus its post wrapped, and the figure
    # is built at exactly the height its panels need.
    for case in drawn:
        for row in case["rows"]:
            row["text_h"] = row["lines"] * line_h + 0.03

    header_h = TOP_MARGIN

    total_h = (header_h + BOTTOM_MARGIN
               + sum(CASE_TITLE_H + CASE_GAP
                     + sum(row["text_h"] + METHOD_GAP
                           for row in case["rows"])
                     for case in drawn))

    fig = plt.figure(figsize=(FIG_WIDTH, total_h))

    def fx(inches: float) -> float:
        return inches / FIG_WIDTH

    def fy(from_top: float) -> float:
        """Figure fraction for a distance measured DOWN from the top edge."""
        return (total_h - from_top) / total_h

    body_x = MARGIN_L + GUTTER
    cursor = header_h
    for case in drawn:
        post = case["post"]
        fig.text(fx(MARGIN_L), fy(cursor + CASE_TITLE_H * 0.55),
                 f"{post['id']} · {case['heading']} · "
                 f"{case['confusion'].title()}",
                 fontsize=args.fontsize + 2, fontweight="bold", ha="left",
                 va="center", color=INK)
        cursor += CASE_TITLE_H

        for row in case["rows"]:
            row_top = cursor
            text_h = row["text_h"]
            # The method name sits on the same row as its panel's first
            # line, not above it, so label and sentence read as one line.
            fig.text(fx(MARGIN_L), fy(row_top + line_h * 0.42),
                     display_name(row["method"]), fontsize=args.fontsize,
                     fontweight="bold", ha="left", va="center", color=INK)

            ax = fig.add_axes([fx(body_x), fy(row_top + text_h),
                               fx(panel_w), text_h / total_h])
            draw_panel(ax, fig, row["tokens"], fontsize=args.fontsize,
                       line_step=line_h / text_h, fills=row["fills"],
                       band=row["gold"], band_fill=False)
            cursor = row_top + text_h + METHOD_GAP
        cursor += CASE_GAP

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
