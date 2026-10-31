"""Causal-faithfulness machinery for the fine-tuning arms' word-deletion audit,
shared by src.stage2.word_deletion and the fused training-time audit in
src.stage1.train. Not run directly.

Rankings are always the method's OWN attribution. The frozen clinical-feature
manifest that once supplied them was retired on 2026-08-26 (quarantine:
~/lonxplain_removed_sae_metrics_20260826) after blind re-inspection showed its
members were largely lexical and register detectors rather than the loneliness
construct; the SAE own-feature knockout that replaced it was itself removed on
2026-08-26 (quarantine: ~/lonxplain_removed_saefaith_20260826) as basis-dependent
and, on the probe path, never functional.

1. CURVE HELPERS — word deletion, normalized logit decay, trapezoid AOPC.
2. ARM WORD DELETION (`arm_word_deletion_rows`) — rank the arm's own words,
   delete them from the text, re-score. No SAE, no hook, no dictionary.
3. PROBE FAITHFULNESS — removed; the probes are word-deleted through
   `src.utils.frozen_backbone`, driven by `src.stage2.word_deletion --compute`.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

from src.baselines.fine_tune import build_inputs
from src.utils.data_prep import words
# The hook adapter is the object the arms are audited through; it is not an
# SAE dependency (it wraps the plain PooledClassifier and is parity-checked).
from src.utils.sae_features import PooledClassifierHookAdapter


# ============================================================ 1. curve helpers
"""Shared word-deletion and normalized-decay helpers for the SAE audit.

The ranking these helpers serve is the method's OWN word attribution: the
interventions section below ranks words by the score that method assigns them
— the same ranking `dla_auprc` scores word AUPRC over — builds the deletion
plans and the matched random-word controls, and calls in here to turn those
plans into scored, normalized decay curves. Plausibility and faithfulness
therefore interrogate one ranking, not two.

Plausibility asks whether a model's attributions look like the expert's spans.
Faithfulness asks the different, harder question of whether those attributions
are what the model actually used. v1 could answer it by construction for probes
(the post score *was* the token-count-weighted mean of the word scores) and not
at all for the fine-tuned arms. Attention pooling breaks that identity even for
probes (src/probe/attention_probe), so under the v2 design faithfulness has to
be measured, and this is where hypothesis H2b is decided.

THE THREE CHOICES THAT MAKE THIS HONEST
---------------------------------------
1. What "erase" means. Zeroing input embeddings is the usual shortcut and is
   excluded here: a zero embedding is an input the model never saw in training,
   so the confidence collapse it produces measures out-of-distribution
   brittleness. Stage 1 claims the arms *differ* in exactly that brittleness, so
   using it here would make the faithfulness argument circular. Instead a word
   is erased by re-tokenizing the post without it (score_word_deletions) — a
   shorter real post, which is the closest thing to an in-distribution edit.

2. What the decay is measured against. Raw probability drop rewards whichever
   model was more confident to begin with, which is not faithfulness. Every
   curve is normalized by that model's own clean-to-empty logit range, so the
   quantity is the *fraction of the model's own decision* the erased words
   carried, and the two models are on one scale.

3. What counts as evidence. Erasing high-scoring words lowers the score in any
   model, because erasing anything does. Every top-k point is therefore paired
   with random-k erasures of the same size on the same post, and faithfulness is
   the MARGIN between them. A model whose top-k does no better than its random-k
   has attributions that point nowhere, however plausible they look.

The "empty" endpoint is defined, not measured by feeding an empty string: with
every content token gone the pooled vector is 0 by both poolings' own algebra
(the masked mean divides by a clamped zero; the attention softmax over an
all-masked row is undefined and would come back NaN), so the model's score is
exactly its head bias. Taking that as the endpoint also keeps
build_inputs' pathological-truncation fallback out of the denominator — that
branch would silently pool over chat-template tokens and put a number in the
denominator that corresponds to no input at all.
"""





def k_word_count(n_words: int, k_percent: float) -> int:
    return max(1, min(n_words, int(np.ceil(n_words * k_percent / 100.0))))


def delete_words(text: str, post_words: list[dict], drop: set[int]) -> str:
    """The post with `drop` word indices removed, spliced on character offsets."""
    keep, cursor = [], 0
    for i, w in enumerate(post_words):
        if i in drop:
            keep.append(text[cursor : w["start"]])
            cursor = w["end"]
    keep.append(text[cursor:])
    return "".join(keep)


def normalized_logit_decay(
    clean_logit: float,
    intervened_logit: float,
    empty_logit_value: float,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Fraction of the model's clean-to-empty logit gap removed."""
    denominator = clean_logit - empty_logit_value
    if abs(denominator) <= epsilon:
        return float("nan")
    return float((clean_logit - intervened_logit) / denominator)


def normalized_aopc(
    k_grid: list[float] | np.ndarray,
    normalized_decay: list[float] | np.ndarray,
) -> float:
    """Trapezoidal mean decay over the preregistered perturbation grid.

    The clean point ``(K=0, decay=0)`` is inserted, and the trapezoid is
    divided by the covered x-range, so the statistic is the mean height of the
    piecewise-linear interpolant on ``[0, max(K)]``. Because the pinned grid is
    geometric (1..50), that mean is dominated by the widest interval — the gap
    between the top two grid points — rather than weighting each K equally.

    This is NOT comparable across grids of different length: a curve pinned at
    decay 1.0 everywhere scores 0.99 on the six-point grid and 0.50 on a
    single-point diagnostic grid, because the inserted (0, 0) point occupies
    the whole interval in the latter. Compare AOPC only across arms sharing
    ``faithfulness.k_grid``. Non-finite points produce NaN rather than being
    silently omitted.
    """
    x = np.asarray(k_grid, dtype=float)
    y = np.asarray(normalized_decay, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("k_grid and normalized_decay must be equal non-empty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("k_grid must be strictly increasing and positive")
    x = np.r_[0.0, x]
    y = np.r_[0.0, y]
    return float(np.trapezoid(y, x) / x[-1])


def deterministic_rng(seed: int, text: str) -> np.random.Generator:
    digest = sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def empty_logit(model) -> float:
    """The score of a post with no content: pooled = 0, so the head's bias."""
    return float(model.head.bias.detach().float().cpu())


@torch.no_grad()
def _score_texts(model, tok, texts: list[str], max_len: int,
                 batch_size: int = 16) -> np.ndarray:
    # No variants to score is a legitimate state, not an error: a post on which
    # no clinical feature fires produces an empty deletion plan. Return an empty
    # array rather than letting np.concatenate([]) raise. Checked before the
    # model is touched, so the empty case needs no model at all.
    if not len(texts):
        return np.empty(0, dtype=float)
    device = model_device(model)
    out = []
    for i in range(0, len(texts), batch_size):
        ids, attn, post = build_inputs(tok, texts[i : i + batch_size], max_len)
        out.append(
            model(ids.to(device), attn.to(device), post.to(device))
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(out)


def score_word_deletions(
    model,
    tokenizer,
    text: str,
    drop_sets: list[set[int]],
    *,
    max_len: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Re-tokenize and score natural-language word-deletion interventions."""
    post_words = words(text)
    variants = [delete_words(text, post_words, drop) for drop in drop_sets]
    return _score_texts(model, tokenizer, variants, max_len, batch_size)


# ================================================== 2. the arm word-deletion audit
#
# The SAE own-feature knockout was removed on 2026-08-26 (pre-removal copies in
# ~/lonxplain_removed_saefaith_20260826) as basis-dependent, and the activation
# patching that briefly replaced it was dropped from the study on 2026-08-26.
# What survives here is the fine-tuning arms' word-deletion audit, which never
# needed a dictionary: it ranks the arm's own words, deletes them from the text
# and re-scores.

def _add_aopc(rows: list[dict], k_grid: list[float]) -> None:
    grouped: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["k_percent"] > 0:
            grouped[(row["intervention"], row["control"])][
                row["k_percent"]
            ].append(row["normalized_decay"])
    values = {}
    for key, by_k in grouped.items():
        mean_curve = [
            float(np.mean(by_k[k])) if k in by_k else float("nan")
            for k in k_grid
        ]
        values[key] = normalized_aopc(k_grid, mean_curve)
    for row in rows:
        row["normalized_aopc"] = values.get(
            (row["intervention"], row["control"]), 0.0
        )

@torch.no_grad()
def arm_word_deletion_rows(
    adapter: PooledClassifierHookAdapter,
    tokenizer,
    text: str,
    word_scores: np.ndarray,
    *,
    config: dict,
    finetune_config: dict,
    seed: int,
    ranking_source: str = "own_attribution",
) -> list[dict]:
    """Word-deletion rows for one fine-tuned cell's post.

    No SAE, no layer, no hook: deletion happens in the text and the arm rescores
    the shortened post, so this audit reads the classifier and nothing else. The
    `sae`/`profile`/`layer` arguments the SAE-knockout version took went away
    with the knockout.

    `word_scores` decides WHICH words are deleted and the arm decides what that
    costs; the two are separable, which is what let a shared external ranking
    be passed here instead of the arm's own. The only caller left is
    `src.stage2.word_deletion`, which always passes the arm's own attribution,
    so `ranking_source` is now always `own_attribution` -- still written on
    every row, because it is the field that would catch a future ranking being
    pooled with these.
    """
    fc = config["faithfulness"]
    interventions = set(fc.get("interventions", ()))
    if interventions != {"word_delete"}:
        raise ValueError(
            f"faithfulness.interventions must be ['word_delete'], "
            f"got {sorted(interventions)}"
        )
    if fc.get("aopc_method") != "trapezoid":
        raise ValueError("faithfulness.aopc_method must be 'trapezoid'")
    controls = set(fc.get("controls", ()))
    if controls != {"random_words"}:
        raise ValueError(
            f"faithfulness.controls must be ['random_words'], got {sorted(controls)}"
        )
    k_grid = [float(k) for k in fc["k_grid"]]
    draws = int(fc["random_draws"])
    rng = deterministic_rng(seed, text)
    max_len = int(finetune_config["max_len"])
    # Deleting nothing is the clean forward, on the identical code path the
    # intervened variants take -- a separately computed clean logit could differ
    # from them by tokenization alone and would bias every decay by that offset.
    clean = float(score_word_deletions(
        adapter.classifier, tokenizer, text, [set()], max_len=max_len)[0])
    empty = empty_logit(adapter.classifier)
    gap = clean - empty
    small_gap = abs(gap) < float(fc.get("small_logit_gap_warn", 0.1))
    common = {
        "clean_logit": clean,
        "empty_logit": empty,
        "clean_empty_logit_gap": gap,
        "small_logit_gap": small_gap,
        "ranking_source": ranking_source,
        "audit_protocol": "own_word_v2",
    }
    rows: list[dict] = []

    scores = np.asarray(word_scores, dtype=float)
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) == 0:
        raise ValueError(
            f"{ranking_source} word scores contain no finite content words")
    # Own word attributions are sparse: most words score exactly 0 and a stable
    # sort would break those ties by document position -- "the top K% words"
    # would silently become "the first K% of the post", and the margin against
    # random deletion would measure position rather than attribution.
    active_words = valid[scores[valid] > 0]
    ranked = active_words[np.argsort(-scores[active_words], kind="stable")]
    n_active_words = int(active_words.size)
    common["n_active_words"] = n_active_words
    common["n_valid_words"] = int(valid.size)
    drop_sets: list[set[int]] = []
    plans: list[tuple[str, float, int, int]] = []
    for k in k_grid:
        # Sized off the active pool, but the random control is drawn from the
        # full valid pool at the SAME count, so the two remain matched on how
        # much text is removed.
        n_selected = min(k_word_count(n_active_words, k), n_active_words)
        if n_selected:
            drop_sets.append(set(ranked[:n_selected].tolist()))
            plans.append(("target", k, n_selected, 0))
            for replicate in range(draws):
                drop_sets.append(
                    set(rng.choice(valid, size=n_selected, replace=False).tolist())
                )
                plans.append(("random_words", k, n_selected, replicate))
        else:
            # No word carries positive own attribution: a no-op here.
            rows.append({
                **common,
                "intervention": "word_delete", "control": "target",
                "k_percent": k, "n_selected": 0, "replicate": 0,
                "intervened_logit": clean,
                "normalized_decay": normalized_logit_decay(clean, clean, empty),
            })
    logits = score_word_deletions(
        adapter.classifier, tokenizer, text, drop_sets, max_len=max_len)
    for (control, k, n_selected, replicate), logit in zip(plans, logits):
        rows.append({
            **common,
            "intervention": "word_delete", "control": control,
            "k_percent": k, "n_selected": n_selected, "replicate": replicate,
            "intervened_logit": float(logit),
            "normalized_decay": normalized_logit_decay(clean, float(logit), empty),
        })

    _add_aopc(rows, k_grid)
    return rows


# ==================================================== 3. the probe faithfulness path
#
# What used to live here was a ~1,500-line SAE engine that encoded each probe's
# residual onto the pinned dictionary and ablated decoder directions. It was
# removed on 2026-08-26 (pre-removal copy in ~/lonxplain_removed_saefaith_20260826)
# for two reasons.
#
# It never worked. `build_post_plan` raised NotImplementedError unconditionally:
# the engine built ONE intervention plan and scored every probe against it, which
# was correct while the ranking came from a shared external manifest and became
# wrong once each method ranked by its own weights. No probe faithfulness cell
# was ever produced by it.
#
# And it was the wrong instrument. Ranking a probe's SAE features and removing
# decoder directions advantages `sae_probe` by construction -- its logit is
# exactly linear in the latents being removed, while every other probe reads the
# raw residual and keeps an unremovable reconstruction-error term the ablation
# cannot touch.
#
# Probe faithfulness is word deletion too: `src.utils.frozen_backbone`
# re-encodes each deleted variant through the backbone (the activation cache
# cannot be reused once the text changes) and scores it with the probe's own
# `score_vectors`, so no dictionary enters the measurement. The fine-tuning
# arms keep `arm_word_deletion_rows` above. Both halves are driven by
# `src.stage2.word_deletion`.
