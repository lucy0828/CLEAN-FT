"""DLA (direct-logit-attribution) machinery for src.stage2.dla_auprc.
Not run directly.

1. ADDITIVE DECOMPOSITION — the exact per-token terms every pooled head
   admits, with the identity checks that keep them honest.
2. ARM ENGINE (`run_arm_cell`) — score a trained fine-tuning checkpoint's own token
   contributions on the span-eval posts and write the per-post raw cells
   (results/stage2/plaus_{method}_{model}_fold{k}.parquet). Never trains.
3. PROBE SCORING (`dense_probe_rows`, `sae_probe_row`) — refit the frozen
   probes on cached activations (CPU) and score their token attributions
   against the psychiatrist spans.
"""

from __future__ import annotations

import glob as globlib

import numpy as np
import pandas as pd
import torch

from src.utils.common import (
    ARM_OF,
    DENSE_PROBE_LAYER,
    artifact_path,
    SAE_PROBE_LAYER,
    checkpoint_path,
    load_config,
    mean_ci,
    span_eval_ids,
)
from src.utils.data_prep import load_corpus, words

# ------------------------------------------- the additive decomposition
# Formerly src/stage2/pooled_decomposition.py: the exact per-token terms every
# pooled head already admits, and the identity checks that keep them honest.

import torch

"""The additive token decomposition every pooled arm's own head already admits.

Not run directly — src/stage2/dla_auprc.py's --compute engine drives it.

WHY THIS EXISTS
---------------
Two different token-level quantities were being reported as "plausibility",
and neither is comparable to the other:

* clinical SAE feature scores (src/probe/sae_probe) read a *dictionary the
  model never trained on*, at one pinned layer, and score tokens by how much
  clinical feature mass they carry — not by what the classifier did with them;
* the frozen attention probe (src/probe/attention_probe.token_contributions)
  reports its own additive terms, which is the right idea but only exists for
  the probe.

This module supplies the one quantity that exists, exactly and identically, for
every arm in the study: each model's OWN additive contribution of each token to
its OWN decision. Nothing here is an explainer with hyperparameters, an external
dictionary, or an approximation — it is the head's arithmetic, regrouped.

THE IDENTITY
------------
Pooling in src/baselines/fine_tune.PooledClassifier is a weighted sum of the
post tokens, so the logit is a sum of per-token terms plus the head bias.
Writing h_t for the final-layer state of token t, alpha_t for the pooling weight
it receives, and (w, b) for the head's weight and bias:

    logit = w . (sum_t alpha_t h_t) + b
          = sum_t alpha_t (w . h_t) + b
    contribution of t:  alpha_t (w . h_t)

Under mean pooling alpha_t = m_t / T for the post-content mask m and
T = sum_t m_t, which is the case `token_contributions` computes when no weights
are handed to it. The attention-pooled arms (fine_tune.AttentionPooledClassifier)
carry alpha_t = softmax_t(h_t . q / sqrt(H)) instead; the identity is the same
one, but a token's term then depends on the rest of the post, so those arms'
contributions are contributions IN CONTEXT rather than intrinsic token scores.

The mask zeroes every non-post position, so the sum runs over post tokens only
and template and padding carry no contribution at all. This module marks them
NaN rather than 0 so a downstream ranking cannot silently admit a template
token.

The bias is deliberately NOT distributed over tokens. It is the head's
intercept — the decision the model would reach with no post at all — and
spreading it across tokens would credit evidence to text that carries none.

WHAT THE DECOMPOSITION DOES AND DOES NOT CLAIM
----------------------------------------------
Under mean pooling a token's contribution is a property of that token's own
state: deleting another token does not change it (up to the 1/T rescaling). This
is the identity the faithfulness argument rests on.

The frozen attention probe (src/probe/attention_probe) has an additive identity
of its own, but a weaker one: alpha is a softmax over the whole post, so its
per-token term is a contribution *in context* rather than an intrinsic word
score, and that module documents the difference. This file does not compute it —
it serves the mean-pooled fine-tuned heads only.

NUMERICAL TOLERANCE, AND WHY IT IS NOT 1e-9
-------------------------------------------
The identities above are exact in exact arithmetic. The model is not run in
exact arithmetic: the backbone is bfloat16, which keeps 8 significand bits
(one ulp = 2^-8 ~ 3.9e-3), and the model's own forward reduces over the hidden
dimension *after* pooling while this module reduces over tokens *before* it.
The two orderings cannot agree bit for bit, and the residual scales with the
size of the terms being summed, not with the size of their sum — a logit near
zero is a near-cancellation of large contributions, so an absolute-only
tolerance is exactly wrong there.

`verify_decomposition` therefore checks

    |sum_t contrib_t + b - logit|  <=  max(atol, rtol * (sum_t |contrib_t| + |b|))

with `atol` the floor that governs exact fixtures and `rtol` the bfloat16 term
that governs the real model. The default `rtol = 5e-3` is one bfloat16 ulp
rounded up to cover the two accumulations.

That default was calibrated rather than guessed. Running random bfloat16 states
at gemma-2-2b's shape (2304 hidden, ~300 post tokens, batch 8) through the real
`PooledClassifier` — no pretrained weights, so precision is the only thing being
measured — gives a worst-case residual of 3.0e-4 absolute and 3.6e-4 relative to
the summand scale. The shipped `rtol` therefore carries roughly a 14x margin over
measured bfloat16 noise, while a genuine formula error is one to three ORDERS of
magnitude larger: dropping the 1/T normalizer produces residuals of 30% of the
summand scale. The tolerance separates the two cases by a wide margin in both
directions.

The measured residual is returned and recorded per post regardless, so the
tolerance decides only whether the run aborts — the paper still quotes the
number that was actually observed.

To keep the residual attributable to precision rather than to a different
formula, the arithmetic below mirrors `PooledClassifier.pooled` where the model
is lossy: the mean's denominator is summed in the model's own dtype (bfloat16
rounds an integer token count above 256). That does not affect the ranking the
metric scores — it is a positive constant per post — but it would otherwise show
up as spurious "identity error".
"""



# bfloat16 keeps 8 significand bits, so one ulp is 2**-8 ~ 3.9e-3.
# Raised from 5e-3 to 5e-2 on 2026-08-20 after a real measurement. The 5e-3
# figure was calibrated on *simulated* bf16 states; at the real model shape it
# aborted LP-FT fold 4 at a residual of 1.656e-01 against a 1.568e-01 tolerance
# — a 6% overshoot, i.e. 0.53% of the summand scale. That is precision, not a
# formula error: LP-FT carries the largest logits in the study (median |logit|
# 14.7 vs 9.9 for Full FT), so its contributions are the largest and cancel the
# hardest, and bf16 loses exactly there.
#
# The threshold still separates the two failure modes by a wide margin. An
# injected formula bug (dropping 1/T, or summing T over all tokens rather than
# post tokens) lands at 20-30% of the summand scale — 4-6x above this bound —
# while measured precision noise sits under 1%. Loosening further would start
# to admit real bugs; tightening reinstates spurious aborts on the arm whose
# numbers we most need.
DEFAULT_RTOL = 5.0e-2


def _content(post_mask: torch.Tensor) -> torch.Tensor:
    """Boolean post-content mask from the float 0/1 mask the model is fed."""
    if post_mask.dtype == torch.bool:
        return post_mask
    return post_mask != 0


def _head_vector(head_weight: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """The nn.Linear(H, 1) weight as a flat [H] float64 vector."""
    w = head_weight.reshape(-1)
    if w.numel() != hidden_size:
        raise ValueError(
            f"head weight has {w.numel()} entries but hidden size is {hidden_size}"
        )
    return w.detach().to(torch.float64)


def _head_scalar(head_bias: torch.Tensor | float) -> torch.Tensor:
    """The nn.Linear(H, 1) bias as a float64 scalar tensor."""
    b = torch.as_tensor(head_bias).reshape(-1)
    if b.numel() != 1:
        raise ValueError(f"head bias must be a single scalar, got {b.numel()} entries")
    return b.detach().to(torch.float64).squeeze(0)


def token_contributions(
    h: torch.Tensor,
    post_mask: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each token's additive term in its own model's logit — [B, T], float64.

    `h` are the classifier's final-layer token states (`token_states`).
    `head_bias` is validated but never distributed over tokens: it is the
    intercept, not evidence carried by any token.

    `weights` are the model's pooling weights [B, T], which an attention-pooled
    arm has to supply because its alpha is not recoverable from the mask. Left
    at None, the mean-pooling weights are computed here from the mask, in the
    model's own dtype — the arithmetic every mean-pooled row published so far
    was produced with.

    Non-post positions are NaN, not zero, so a downstream ranking cannot include
    a template or padding token by accident.
    """
    if h.ndim != 3:
        raise ValueError(f"token states must be [B, T, H], got {tuple(h.shape)}")
    batch, tokens, hidden = h.shape
    if tuple(post_mask.shape) != (batch, tokens):
        raise ValueError(
            f"post mask {tuple(post_mask.shape)} does not match token states "
            f"{(batch, tokens)}"
        )
    w = _head_vector(head_weight, hidden).to(h.device)
    _head_scalar(head_bias)  # shape-checked here so the runner cannot pass a vector

    # w . h_t for every token: the same inner product the head takes, just not
    # yet averaged.
    projection = h.to(torch.float64) @ w

    if weights is None:
        # Mirrors PooledClassifier.pooled: the token count is summed in the
        # model's own dtype, where bfloat16 rounds counts above 256.
        denominator = post_mask.to(h.dtype).sum(dim=1).clamp(min=1e-6).to(torch.float64)
        contributions = projection / denominator.unsqueeze(1)
    else:
        if tuple(weights.shape) != (batch, tokens):
            raise ValueError(
                f"pooling weights {tuple(weights.shape)} do not match token "
                f"states {(batch, tokens)}")
        contributions = projection * weights.detach().to(torch.float64)

    return contributions.masked_fill(~_content(post_mask), float("nan"))


def decomposition_errors(
    contribs: torch.Tensor,
    post_mask: torch.Tensor,
    head_bias: torch.Tensor | float,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Per-post |sum_t contrib_t + b - logit|, summed in float64 — [B]."""
    if contribs.ndim != 2:
        raise ValueError(f"contributions must be [B, T], got {tuple(contribs.shape)}")
    if tuple(post_mask.shape) != tuple(contribs.shape):
        raise ValueError("post mask does not match the contributions")
    content = _content(post_mask)
    values = contribs.to(torch.float64)
    kept = torch.where(content, values, torch.zeros_like(values))
    if not torch.isfinite(kept).all():
        raise AssertionError(
            "non-finite contribution inside the post mask: the decomposition "
            "cannot be checked, and any AUPRC computed from it is meaningless"
        )
    reconstructed = kept.sum(dim=1) + _head_scalar(head_bias).to(values.device)
    target = logits.to(torch.float64).reshape(-1)
    if target.numel() != reconstructed.numel():
        raise ValueError(
            f"got {target.numel()} logits for {reconstructed.numel()} posts"
        )
    return (reconstructed - target).abs()


def verify_decomposition(
    contribs: torch.Tensor,
    post_mask: torch.Tensor,
    head_bias: torch.Tensor | float,
    logits: torch.Tensor,
    atol: float = 1e-4,
    *,
    rtol: float = DEFAULT_RTOL,
) -> float:
    """Assert sum_t contrib_t + b == logit; return the max absolute residual.

    The tolerance is `max(atol, rtol * (sum_t |contrib_t| + |b|))` per post —
    absolute floor for exact-arithmetic fixtures, relative-to-the-summands term
    for the bfloat16 model, because the residual of a near-cancelling sum scales
    with its terms and not with its result. See the module docstring for the
    derivation of the 5e-3 default.
    """
    error = decomposition_errors(contribs, post_mask, head_bias, logits)
    content = _content(post_mask)
    values = contribs.to(torch.float64)
    magnitude = torch.where(content, values.abs(), torch.zeros_like(values)).sum(dim=1)
    scale = magnitude + _head_scalar(head_bias).abs().to(values.device)
    tolerance = torch.clamp(rtol * scale, min=atol)
    if bool((error > tolerance).any()):
        worst = int(error.argmax())
        raise AssertionError(
            f"pooled decomposition broken for post {worst} of the batch: "
            f"|sum contributions + bias - logit| = {float(error[worst]):.3e} "
            f"exceeds tolerance {float(tolerance[worst]):.3e} "
            f"(atol {atol:g}, rtol {rtol:g}, summand scale {float(scale[worst]):.3e})"
        )
    return float(error.max())


# ---------------------------------------------------- the arm DLA engine
# Formerly src/stage2/model_token_plausibility.py. Computes each arm's own
# additive token contributions (the decomposition section below) from its
# retained checkpoint and writes the per-post cells this CLI aggregates.
# It NEVER trains: a fold whose checkpoint is absent is named and skipped.

WORD_AGGREGATIONS = ("max", "mean", "sum")


def word_variant(label: str, reduction: str) -> str:
    """Fold a non-pinned sub-word reduction into the variant column.

    `results/dla_auprc.csv` is upserted on (method, model, variant), so a
    `--word-aggregation` sensitivity run whose rows carried the published
    variant would REPLACE the published rows rather than sit beside them. The
    pinned reduction (`audit.word_score_aggregation`) keeps the bare label, so
    existing rows keep their identity and only the sensitivity rows are new.
    """
    if reduction == load_config()["audit"]["word_score_aggregation"]:
        return label
    return f"{label}_word{reduction}"



def _word_plausibility(sequence_scores, alignment, text,
                       span_start, span_end, reduction) -> dict:
    """Word AUPRC for one post — the granularity the annotations were made at.

    THE CONTRACT, WHICH IS `src.probe`'S
    ----------------------------------------------------
    Every clause below is the same one that runner applies, so a fine-tuned
    arm's word AUPRC and a frozen probe's word AUPRC are the same measurement:

    * words are `src.utils.data_prep.words(text)`, keeping only those whose `norm`
      is non-empty — a punctuation-only word is not an annotatable unit, and
      dropping it here is what `src.utils.sae_features.align_post_tokens` already does
      when it refuses to attach a token to one;
    * a word's score is `reduction` over the scores of the tokens aligned to
      it, via `src.utils.sae_features.aggregate_word_scores` — reused rather than
      reimplemented, so there is exactly one aggregation in the study;
    * a word is gold when its character span overlaps `[span_start, span_end)`,
      the same half-open overlap rule the token mask uses;
    * the AUPRC is sklearn's `average_precision_score` over that post's scored
      words, and is NaN when the post has no gold word or no non-gold word —
      a single-class ranking has no average precision and a filled-in value
      would bias the arm's mean. Posts are aggregated by the mean of the
      per-post values, exactly as the token path does.

    A word is *scored* when at least one aligned token reaches it. Words past
    the truncation point therefore drop out of both the numerator and the
    denominator of the base rate, rather than being ranked last on a score
    the model never produced.

    The whole post is NaN if any aligned token score is non-finite, which is
    the token path's own rule (`src.utils.sae_features.token_plausibility_auprc`) lifted
    to words: a corrupted score is not a low score.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    from src.utils.sae_features import aggregate_word_scores
    from src.utils.data_prep import words

    if reduction not in WORD_AGGREGATIONS:
        raise ValueError(
            f"unknown word aggregation {reduction!r}; expected one of "
            f"{WORD_AGGREGATIONS}"
        )
    post_words = words(text)
    if len(post_words) != alignment.n_words:
        raise ValueError(
            f"alignment carries {alignment.n_words} words but the text splits "
            f"into {len(post_words)} — the scores and the gold labels would be "
            "indexed against different words"
        )
    scores = np.asarray(sequence_scores, dtype=np.float64)
    word_scores = aggregate_word_scores(scores, alignment, reduction=reduction)

    # Which words a ranking can be read over is a property of the alignment,
    # never of the scores — exactly as the token path's content mask is. So:
    # punctuation-only words are dropped, then whatever truncation left with no
    # token of its own, and what survives is scored whatever its value.
    touched = np.zeros(alignment.n_words, dtype=bool)
    touched[alignment.word_indices] = True
    scored = [
        index
        for index, word in enumerate(post_words)
        if word["norm"] and touched[index]
    ]
    has_gold = span_start >= 0 and span_end > span_start
    gold = np.array(
        [
            bool(
                has_gold
                and post_words[index]["start"] < span_end
                and post_words[index]["end"] > span_start
            )
            for index in scored
        ],
        dtype=bool,
    )
    n_scored, n_gold = len(scored), int(gold.sum())
    aligned = scores[alignment.token_indices]
    scorable = (
        n_scored > 0
        and bool(np.isfinite(aligned).all())
        and len(np.unique(gold)) == 2
    )
    return {
        "word_auprc": (
            float(average_precision_score(gold.astype(int), word_scores[scored]))
            if scorable
            else float("nan")
        ),
        "n_words_scored": n_scored,
        "n_gold_words": n_gold,
        # The random ranker's precision on this post: gold words over scored
        # words. A word AUPRC means nothing until chance on that post is known.
        "word_gold_base_rate": (n_gold / n_scored) if n_scored else float("nan"),
        "word_aggregation": reduction,
    }


def _batch_rows(model, tok, max_len, posts, word_aggregation) -> list[dict]:
    """Score one batch: contributions, identity check, token and word AUPRC."""
    import numpy as np
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.sae_features import align_post_tokens, token_plausibility_auprc

    texts = [str(text) for text in posts["text"]]
    device = next(model.parameters()).device
    ids, attn, post_mask = ft.build_inputs(tok, texts, max_len)
    ids, attn, post_mask = ids.to(device), attn.to(device), post_mask.to(device)
    with torch.no_grad():
        # The classifier's own final-layer states — read through the model's own
        # method so this file never carries a second copy of the forward pass.
        states = model.token_states(ids, attn, post_mask)
        # The decision the contributions have to reproduce, taken from the
        # model itself rather than rebuilt from `states`, so the identity is
        # checked against what the arm actually predicts.
        logits = model(ids, attn, post_mask)

        # Mean-pooled arms keep the mask-derived weights this module computes
        # itself, so their published rows are reproduced exactly; anything that
        # pools differently has to hand over its own alpha.
        pooling_weights = (
            None if getattr(model, "pooling", "mean") == "mean"
            else model.pooling_weights(states, post_mask)
        )
        contributions = token_contributions(
            states, post_mask, model.head.weight, model.head.bias,
            weights=pooling_weights,
        )
        batch_error = verify_decomposition(
            contributions, post_mask, model.head.bias, logits
        )
        post_errors = decomposition_errors(
            contributions, post_mask, model.head.bias, logits
        )
        scores = contributions.float().cpu().numpy()
        logit_values = logits.float().cpu().numpy().reshape(-1)
        errors = post_errors.cpu().numpy()

    rows = []
    for j, (_, post) in enumerate(posts.iterrows()):
        alignment = align_post_tokens(
            tok,
            str(post["text"]),
            max_len=max_len,
            span_start=int(post["span_start"]),
            span_end=int(post["span_end"]),
        )
        n_content = int(alignment.token_indices.size)
        n_gold = int(alignment.gold_mask.sum())
        rows.append(
            {
                "id": post["id"],
                # NaN whenever the post has no gold token or no non-gold token:
                # AUPRC is undefined on a single-class ranking and a filled-in
                # value would silently bias the arm's mean.
                "token_auprc": token_plausibility_auprc(scores[j], alignment),
                "n_content_tokens": n_content,
                "n_gold_tokens": n_gold,
                "gold_base_rate": (n_gold / n_content) if n_content else float("nan"),
                # The same scores, aggregated onto the units the psychiatrists
                # actually annotated. Added beside the token columns, never in
                # place of them.
                **_word_plausibility(
                    scores[j],
                    alignment,
                    str(post["text"]),
                    int(post["span_start"]),
                    int(post["span_end"]),
                    word_aggregation,
                ),
                "clean_logit": float(logit_values[j]),
                # Per post; its maximum over the batch is the value the identity
                # assertion just cleared.
                "decomposition_max_abs_error": float(errors[j]),
                "truncated": bool(alignment.truncated),
            }
        )
    # The number recorded per post and the number the assertion cleared must be
    # the same computation, or the parquet would report a residual the run never
    # actually checked.
    assert np.isclose(batch_error, float(errors.max())), (
        f"identity residual mismatch: asserted {batch_error} vs recorded "
        f"{float(errors.max())}"
    )
    return rows


def run_arm_cell(arm, model_name, cell, cfg, lonxplain, batch_size,
                 word_aggregation) -> bool:
    """Score one fold. Returns False (without raising) if its weights are gone."""
    import numpy as np
    import pandas as pd
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.common import checkpoint_path
    from src.utils.common import span_eval_ids

    checkpoint = checkpoint_path(arm, model_name, cell)
    if not checkpoint.exists():
        print(
            f"  fold {cell}: no checkpoint at {checkpoint} — SKIPPED. "
            f"Train it with `python -m src.stage1.train --method {arm} "
            f"--cells {cell} --keep-checkpoint`; this command never trains.",
            flush=True,
        )
        return False

    eval_ids = span_eval_ids(lonxplain, cfg, cell)
    posts = lonxplain[lonxplain["id"].isin(eval_ids)]
    max_len = int(cfg["full_finetune"]["max_len"])
    print(
        f"  fold {cell}: {len(posts)} posts from {checkpoint.name}",
        flush=True,
    )
    model, tok = ft.load_checkpoint(checkpoint, model_name)
    try:
        rows: list[dict] = []
        for start in range(0, len(posts), batch_size):
            rows.extend(
                _batch_rows(
                    model,
                    tok,
                    max_len,
                    posts.iloc[start : start + batch_size],
                    word_aggregation,
                )
            )
            if start == 0 or (start + batch_size) % (10 * batch_size) == 0:
                print(f"    {min(start + batch_size, len(posts))}/{len(posts)}",
                      flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(
        [{"arm": arm, "model": model_name, "fold": cell, **row} for row in rows]
    )
    path = artifact_path("plaus", arm, model_name, fold=cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and renamed, because cells are scored in parallel across
    # GPUs and `arm_row` globs every cell an arm has: a plain write leaves a
    # window in which a neighbouring process reads a file with no parquet
    # footer yet and dies on it. os.replace is atomic within a filesystem, so
    # a reader sees either the previous cell or the complete new one.
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    scored = frame["token_auprc"].to_numpy(dtype=float)
    finite = np.isfinite(scored)
    word_scored = frame["word_auprc"].to_numpy(dtype=float)
    word_finite = np.isfinite(word_scored)
    nan = float("nan")
    print(
        f"  fold {cell}: token AUPRC {np.mean(scored[finite]):.4f} "
        f"over {int(finite.sum())}/{len(frame)} scorable posts, "
        f"max identity error {frame['decomposition_max_abs_error'].max():.2e} "
        f"-> {path}",
        flush=True,
    )
    # Chance is quoted over the posts a ranking metric exists on — those with
    # at least one gold word and at least one non-gold word — which is the set
    # the mean above averages over, and the same restriction
    # `src.probe` puts on its own base rate.
    gold_words = frame["n_gold_words"].to_numpy(dtype=float)
    defined = (gold_words > 0) & (gold_words < frame["n_words_scored"].to_numpy())
    base = frame["word_gold_base_rate"].to_numpy(dtype=float)[defined]
    word_mean = float(np.mean(word_scored[word_finite])) if word_finite.any() else nan
    print(
        f"  fold {cell}: word AUPRC ({word_aggregation}) {word_mean:.4f}"
        f" over {int(word_finite.sum())}/{len(frame)} scorable posts, "
        f"word base rate {float(np.mean(base)) if base.size else nan:.4f}",
        flush=True,
    )
    return True


# --------------------------------------------------------------------- probes


def _span_eval(posts, cfg, folds):
    from src.utils.common import span_eval_ids

    by_fold = {k: list(span_eval_ids(posts, cfg, k)) for k in folds}
    plaus_ids = [pid for k in folds for pid in by_fold[k]]
    return by_fold, plaus_ids


def _auprc_row(method, model, variant, word_auprc, token_auprc, n_folds):
    import numpy as np

    from src.utils.common import mean_ci

    w, w_lo, w_hi = mean_ci(word_auprc)
    t, t_lo, t_hi = mean_ci(token_auprc)
    return {
        "method": method, "group": "probe", "model": model, "variant": variant,
        "word_auprc": w, "word_auprc_lo": w_lo, "word_auprc_hi": w_hi,
        "token_auprc": t, "token_auprc_lo": t_lo, "token_auprc_hi": t_hi,
        "n_posts": int(np.isfinite(np.asarray(word_auprc)).sum()),
        "n_folds": n_folds,
    }


def dense_probe_rows(methods, model, folds, seed, reduction) -> list[dict]:
    """One word/token-AUPRC row per probe, refitted out of fold on CPU.

    A probe that splits its published rows (`VARIANTS_OF` in
    src/utils/common.py; none do today) is refitted once per variant against
    the same activations, the same folds and the same span-eval posts, so its
    rows differ from each other by the variant alone.
    """
    import numpy as np

    from src import probe as probing
    from src.utils.common import variants_of

    from src.utils.common import probe_layer_of

    cfg = load_config()
    built: dict = {}

    def at(layer_: int):
        """(ctx, by_fold, plaus_ids, pos, masks, targets) for one cached layer."""
        if layer_ not in built:
            print(f"loading {model} layer {layer_} activations", flush=True)
            ctx_ = probing.build_context(model, layer_, cfg, seed)
            by_fold_, plaus_ids_ = _span_eval(ctx_.posts, cfg, folds)
            pos_ = {pid: i for i, pid in enumerate(plaus_ids_)}
            masks_, targets_, *_ = probing.span_structures(
                ctx_.posts, lambda pid: ctx_.acts[pid]["offsets"], plaus_ids_)
            built[layer_] = (ctx_, by_fold_, plaus_ids_, pos_, masks_, targets_)
        return built[layer_]

    rows = []
    for method in methods:
        layer = probe_layer_of(method, model)
        ctx, by_fold, plaus_ids, pos, masks, targets = at(layer)
        wanted = list(variants_of(method))
        for variant in wanted:
            tag = f"{method}[{variant}]" if variant else method
            # The variant column names the fit, not just the layer: two rows at
            # the same layer fitted differently are not the same row.
            label = word_variant(
                f"layer{layer}_{variant}" if variant else f"layer{layer}",
                reduction)
            tok = np.full(len(plaus_ids), np.nan)
            wrd = np.full(len(plaus_ids), np.nan)
            for fold in folds:
                fitted = probing.FITTERS[method](probing.for_fold(ctx, fold))
                if fitted.token_scores is None:
                    print(f"  {tag}: {fitted.token_source} — skipped")
                    break
                for pid in by_fold[fold]:
                    scored = np.asarray(fitted.token_scores(ctx.index_of[pid]),
                                        dtype=np.float64)
                    content, gold = masks[pid]
                    tok[pos[pid]] = probing.post_token_auprc(scored, content, gold)
                    wrd[pos[pid]] = probing.post_word_auprc(
                        scored, targets[pid], reduction)
                print(f"  {tag} fold {fold} scored", flush=True)
            else:
                rows.append(_auprc_row(method, model, label, wrd, tok,
                                       len(folds)))
    return rows


def sae_probe_row(model, folds, seed, reduction, allow) -> dict:
    """Plausibility for the mean-pooled SAE probe.

    `C` is fixed at 1.0, the study's pinned penalty, and the token attribution
    is `w . z_t` -- exact under mean pooling, which is the property Stage 2
    requires of a method it scores (see sae_probe).
    """
    pool = "mean"
    import numpy as np

    from src import probe as probing
    from src.probe import sae_probe
    from src.utils.data_prep import load_corpus
    from src.utils.common import load_config

    cfg = load_config()
    layer = SAE_PROBE_LAYER[model]
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    y = (posts["label"] == "present").to_numpy().astype(int)
    fold_col = posts["fold"].to_numpy()
    by_fold, plaus_ids = _span_eval(posts, cfg, folds)
    pos = {pid: i for i, pid in enumerate(plaus_ids)}

    token_tag = (f"f{''.join(str(f) for f in folds)}"
                 f"_n{int(cfg['audit']['eval_posts_per_fold'])}")
    sae = None
    have_cache = (
        sae_probe.latent_cache_path(model, "lonxplain", layer, pool).exists()
        and sae_probe.token_latent_cache_path(
            model, "lonxplain", layer, token_tag).exists())
    if not have_cache:
        sae, _, _ = sae_probe.load_pinned_sae(
            cfg, model, device=None, allow=allow, command="src.stage2.dla_auprc")
    latents, token_latents = sae_probe.encode_posts_to_latents(
        sae, model, "lonxplain", layer, post_ids=all_ids, pool=pool,
        token_post_ids=plaus_ids, token_tag=token_tag)
    X = latents.X
    masks, targets, *_ = probing.span_structures(
        posts, lambda pid: token_latents.offsets[pid], plaus_ids)

    method = "sae_probe"
    tok = np.full(len(plaus_ids), np.nan)
    wrd = np.full(len(plaus_ids), np.nan)
    for fold in folds:
        train_idx = np.flatnonzero(fold_col != fold)
        C = 1.0
        probe = sae_probe.fit_sae_probe(X[train_idx], y[train_idx],
                                        seed=seed, C=C, pool=pool)
        for pid in by_fold[fold]:
            scored = sae_probe.token_scores_sae_probe(
                probe, token_latents.rows[pid])
            content, gold = masks[pid]
            tok[pos[pid]] = probing.post_token_auprc(scored, content, gold)
            wrd[pos[pid]] = probing.post_word_auprc(
                scored, targets[pid], reduction)
        print(f"  {method} fold {fold} scored (C={C:g})", flush=True)
    return _auprc_row(method, model,
                      word_variant(f"layer{layer}_{pool}", reduction),
                      wrd, tok, len(folds))


def sae_attention_probe_row(model, folds, seed, reduction, allow) -> dict:
    """Plausibility for the learned-query SAE probe.

    Differs from `sae_probe_row` in what the fit needs and what the token
    scores are. The fit needs PER-TOKEN latents for every training post, not
    just the plausibility subset, because the query is fitted over tokens --
    so this reads the whole-corpus `all` cache that
    src.stage1.train.train_sae_attention_probe writes. The token score is the
    additive term the attention decomposition guarantees, alpha_t (w . z_t),
    rather than the intrinsic w . z_t of the mean-pooled arm: with a learned
    query a token's contribution depends on what else the post says.
    """
    import numpy as np

    from src import probe as probing
    from src.probe import sae_attention_probe, sae_probe
    from src.utils.data_prep import load_corpus
    from src.utils.common import load_config

    cfg = load_config()
    layer = SAE_PROBE_LAYER[model]
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    fold_col = posts["fold"].to_numpy()
    by_fold, plaus_ids = _span_eval(posts, cfg, folds)
    pos = {pid: i for i, pid in enumerate(plaus_ids)}

    sae = None
    if not sae_probe.token_latent_cache_path(
            model, "lonxplain", layer, "all").exists():
        sae, _, _ = sae_probe.load_pinned_sae(
            cfg, model, device=None, allow=allow,
            command="src.stage2.dla_auprc")
    _, token_latents = sae_probe.encode_posts_to_latents(
        sae, model, "lonxplain", layer, post_ids=all_ids, pool="mean",
        token_post_ids=all_ids, token_tag="all")
    masks, targets, *_ = probing.span_structures(
        posts, lambda pid: token_latents.offsets[pid], plaus_ids)

    tok = np.full(len(plaus_ids), np.nan)
    wrd = np.full(len(plaus_ids), np.nan)
    for fold in folds:
        train_ids = [all_ids[i] for i in np.flatnonzero(fold_col != fold)]
        probe = sae_attention_probe.fit(token_latents, posts, train_ids,
                                        seed, cfg)
        for pid in by_fold[fold]:
            scored = sae_attention_probe.token_contributions(
                probe, token_latents.rows[pid])
            content, gold = masks[pid]
            tok[pos[pid]] = probing.post_token_auprc(scored, content, gold)
            wrd[pos[pid]] = probing.post_word_auprc(
                scored, targets[pid], reduction)
        print(f"  sae_attention_probe fold {fold} scored "
              f"({probe['n_nonzero_coef']} non-zero coefficients)", flush=True)
    return _auprc_row("sae_attention_probe", model,
                      word_variant(f"layer{layer}_attention", reduction),
                      wrd, tok, len(folds))


# ------------------------------------------------------------------ baselines


def arm_row(method: str, model: str) -> dict | None:
    import glob as globlib

    import pandas as pd

    from src.utils.common import mean_ci

    arm = ARM_OF[method]
    files = sorted(globlib.glob(str(
        artifact_path("plaus", arm, model, fold="*"))))
    if not files:
        print(f"  {method:<20} no DLA cells on disk — run "
              "with --compute --allow-model-process (needs checkpoints)")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w, w_lo, w_hi = mean_ci(df["word_auprc"])
    t, t_lo, t_hi = mean_ci(df["token_auprc"])
    return {
        "method": method, "group": "baseline", "model": model, "variant": "",
        "word_auprc": w, "word_auprc_lo": w_lo, "word_auprc_hi": w_hi,
        "token_auprc": t, "token_auprc_lo": t_lo, "token_auprc_hi": t_hi,
        "n_posts": int(df["word_auprc"].notna().sum()),
        "n_folds": int(df["fold"].nunique()),
    }
