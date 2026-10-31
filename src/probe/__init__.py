"""The frozen probing methods and their shared machinery.

Method modules: linear_probe, mlp_probe, attention_probe, attention_mlp_probe,
sae_probe,
afr_probe. `shrunk_attention_probe` is fitted from `attention_probe` with its
query penalized rather than free (L2, selected on inner folds) and lives in
this module rather than one of its own, because it does not change the solver.
This package module carries what the methods share: the out-of-fold fitters
(`build_context`, `FITTERS`, `for_fold`) used by `src.stage1.train` and
`src.stage2.dla_auprc`, and the span scoring of token attributions against
the psychiatrist gold (`token_masks`, `word_targets`, `post_*_auprc`,
`span_structures`), with word level as the pre-registered granularity.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score

from src.probe import (afr_probe, attention_mlp_probe, attention_probe,
                       linear_probe, mlp_probe)
from src.utils.data_prep import load_corpus
from src.utils.data_prep import load_layer, words


@functools.cache
def _sae_profile():
    """`src.probe.sae_probe`, imported on first use (pulls the SAE stack)."""
    from src.probe import sae_probe
    return sae_probe


# ================================================================ 2. fitting


@dataclass
class Fitted:
    """One method fitted on one fold's training posts.

    `post_scores` takes held-out integer indices into the fixed post order;
    `token_scores` takes one post's index and returns one score per cached
    token, or is None when the method has no additive token decomposition.
    """

    post_scores: Callable[[np.ndarray], np.ndarray]
    token_scores: Callable[[int], np.ndarray] | None
    token_source: str
    # Score an ARBITRARY [T, H] token matrix, not one the corpus contains.
    # `post_scores` indexes the fixed post order, which is enough for every
    # metric that only reads posts as encoded; word deletion re-encodes the
    # shortened text into fresh token vectors, so it needs the readout as a
    # function. None where a method cannot score a bare token matrix.
    score_vectors: Callable[[np.ndarray], float] | None = None
    # `token_scores` for an arbitrary [T, H] matrix: one score per row, the
    # same decomposition `token_scores` returns, on tokens the corpus does not
    # contain. This is what an out-of-domain deployment needs -- there is no
    # cached post to index into -- and it stands to `token_scores` exactly as
    # `score_vectors` stands to `post_scores`. None where the method has no
    # additive token decomposition.
    token_vectors_scores: Callable[[np.ndarray], np.ndarray] | None = None
    # Per-FOLD provenance, for methods whose fit differs fold to fold in a way
    # a reader has to see (`afr_probe`: how hard its reweighting actually
    # pushed). Variant-level provenance travels separately, in the callers.
    provenance: dict | None = None


@dataclass
class FoldContext:
    """Everything a fitter needs, built once per backbone/layer."""

    acts: dict
    posts: object
    all_ids: list
    index_of: dict
    y: np.ndarray
    fold_col: np.ndarray
    post_vectors: np.ndarray
    cfg: dict
    seed: int
    layer: int
    train_idx: np.ndarray | None = None


def build_context(model: str, layer: int, cfg: dict, seed: int) -> FoldContext:
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    acts = load_layer(model, "lonxplain", layer)
    missing = [p for p in all_ids if p not in acts]
    if missing:
        raise SystemExit(
            f"{len(missing)} posts have no cached tokens at {model} layer "
            f"{layer} (first: {missing[0]}); run src.utils.data_prep --encode."
        )
    return FoldContext(
        acts=acts,
        posts=posts,
        all_ids=all_ids,
        index_of={pid: i for i, pid in enumerate(all_ids)},
        y=(posts["label"] == "present").to_numpy().astype(np.float64),
        fold_col=posts["fold"].to_numpy(),
        post_vectors=mean_post_vectors(acts, all_ids),
        cfg=cfg,
        seed=seed,
        layer=layer,
    )


def mean_post_vectors(acts: dict, post_ids) -> np.ndarray:
    """[N, H] mean-pooled post vectors, in the order of `post_ids`."""
    return np.stack([
        np.asarray(acts[pid]["vecs"], dtype=np.float64).mean(axis=0)
        for pid in post_ids
    ])


def _affine_of(probe: dict, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """(a, b) with the logistic probe's score equal to v @ a + b for any vector v.

    Flattening sklearn's scaler into the coefficients is what lets one matrix
    multiply score a post's tokens; the assert keeps this in step with
    `linear_probe.score`, so a change there cannot silently desynchronise it.
    """
    scale = probe["scaler"].scale_
    coef = probe["clf"].coef_[0]
    a = coef / scale
    b = float(probe["clf"].intercept_[0] - (probe["scaler"].mean_ / scale) @ coef)
    assert np.allclose(reference @ a + b, linear_probe.score(probe, reference),
                       atol=1e-3), (
        "affine form of the logistic probe disagrees with linear_probe.score"
    )
    return a, b


def fit_linear_probe(ctx: FoldContext) -> Fitted:
    """L2 logistic regression on mean-pooled post vectors.

    Its token scores are the same projection applied to a token instead of a
    post vector — the pooling identity above, so the
    post score is exactly the token-count-weighted mean of the token scores.
    """
    pv, acts, ids = ctx.post_vectors, ctx.acts, ctx.all_ids
    probe = linear_probe.fit(pv[ctx.train_idx],
                                   ctx.y[ctx.train_idx].astype(int),
                                   "logistic", ctx.seed,
                                   C=ctx.cfg["probe"]["logreg_C"])
    a, b = _affine_of(probe, pv[ctx.train_idx][:64])
    return Fitted(
        post_scores=lambda idx: pv[idx] @ a + b,
        token_scores=lambda i: acts[ids[i]]["vecs"].astype(np.float64) @ a + b,
        token_source="linear_probe projection of the token vector",
        score_vectors=lambda V: float(np.asarray(V, dtype=np.float64).mean(0) @ a + b),
        token_vectors_scores=lambda V: np.asarray(V, dtype=np.float64) @ a + b,
    )


def fit_attention_probe(ctx: FoldContext) -> Fitted:
    """A learned free query over the same frozen tokens.

    Token scores are `alpha_t * (w . X_t)`, the additive terms the attention
    decomposition guarantees sum to the post score — contributions in context.
    """
    acts, ids = ctx.acts, ctx.all_ids
    train_ids = [ids[i] for i in ctx.train_idx]
    model = attention_probe.fit(acts, ctx.posts, train_ids, ctx.seed, ctx.cfg)
    # Enforce the identity once per fold rather than trusting it.
    attention_probe.verify_decomposition(model, acts[train_ids[0]])
    return Fitted(
        post_scores=lambda idx: np.array(
            [attention_probe.post_score(model, acts[ids[i]]) for i in idx]),
        token_scores=lambda i: attention_probe.token_contributions(
            model, acts[ids[i]]),
        token_source="attention_probe.token_contributions",
        score_vectors=lambda V: float(
            attention_probe.post_score(model, {"vecs": np.asarray(V)})),
        token_vectors_scores=lambda V: attention_probe.token_contributions(
            model, {"vecs": np.asarray(V)}),
    )


def _attention_cfg(ctx: FoldContext, **overrides) -> dict:
    """`ctx.cfg` with the `attention_pool` block overridden for one arm.

    A shallow copy of the block, not of the config: the two derived attention
    arms differ from `attention_probe` in the query only, and copying the whole
    config would let some other block drift between them without anyone seeing
    it.
    """
    cfg = dict(ctx.cfg)
    cfg["attention_pool"] = {**ctx.cfg["attention_pool"], **overrides}
    return cfg


def _attention_fitted(ctx: FoldContext, model, provenance: dict) -> Fitted:
    """The `Fitted` every attention-pooled arm returns — one decomposition."""
    acts, ids = ctx.acts, ctx.all_ids
    return Fitted(
        post_scores=lambda idx: np.array(
            [attention_probe.post_score(model, acts[ids[i]]) for i in idx]),
        token_scores=lambda i: attention_probe.token_contributions(
            model, acts[ids[i]]),
        token_source="attention_probe.token_contributions",
        score_vectors=lambda V: float(
            attention_probe.post_score(model, {"vecs": np.asarray(V)})),
        token_vectors_scores=lambda V: attention_probe.token_contributions(
            model, {"vecs": np.asarray(V)}),
        provenance=provenance,
    )


def fit_shrunk_attention_probe(ctx: FoldContext) -> Fitted:
    """`attention_probe` with an L2 penalty on the query, chosen on inner folds.

    The one thing it changes is that the query is no longer the only
    unpenalized object in the readout. The penalty is selected by
    `attention_probe.select_q_l2` on folds INSIDE this fold's training pool, so
    the outer held fold never enters the selection and the study's refusal to
    tune on the evaluation fold is intact.

    As the penalty grows the query goes to zero and the arm becomes
    `linear_probe`; at zero it is `attention_probe`. The selected value is on
    every scored row, because "how hard did the query have to be held down"
    is the finding, not an implementation detail.
    """
    acts, ids = ctx.acts, ctx.all_ids
    train_ids = [ids[i] for i in ctx.train_idx]
    cfg = _attention_cfg(ctx, q_l2_select=True)
    model = attention_probe.fit(acts, ctx.posts, train_ids, ctx.seed, cfg)
    attention_probe.verify_decomposition(model, acts[train_ids[0]])
    sel = model["selection"]
    return _attention_fitted(ctx, model, {
        "q_l2_selected": sel["q_l2"],
        "attention_entropy": float(np.mean([
            attention_probe.attention_entropy(model, acts[i])
            for i in train_ids[:256]])),
    })


def fit_attention_mlp_probe(ctx: FoldContext) -> Fitted:
    """The interaction cell: a learned query AND a nonlinear head.

    Exists to answer whether capacity adds anything once the pooling is right
    (src/probe/attention_mlp_probe.py). Its token scores close by
    integrated-gradient completeness rather than by the exact identity
    `attention_probe` keeps, so it is a diagnostic and not a Stage 2 readout.
    """
    acts, ids = ctx.acts, ctx.all_ids
    train_ids = [ids[i] for i in ctx.train_idx]
    model = attention_mlp_probe.fit(
        acts, ctx.posts, train_ids, ctx.seed, ctx.cfg)
    steps = int(ctx.cfg["probe"].get("mlp_ig_steps",
                                     attention_mlp_probe.IG_STEPS))
    # Enforce completeness once per fold rather than trusting it, exactly as
    # the other two inexact probes do.
    attention_mlp_probe.verify_attribution(
        model, [acts[i] for i in train_ids[:16]], steps)
    return Fitted(
        post_scores=lambda idx: np.array(
            [attention_mlp_probe.post_score(model, acts[ids[i]]) for i in idx]),
        token_scores=lambda i: attention_mlp_probe.token_contributions(
            model, acts[ids[i]], steps),
        token_source="attention_mlp_probe.token_contributions",
        score_vectors=lambda V: attention_mlp_probe.score_vectors(model, V),
        token_vectors_scores=lambda V: attention_mlp_probe.token_contributions(
            model, {"vecs": np.asarray(V)}, steps),
    )


def fit_mlp_probe(ctx: FoldContext) -> Fitted:
    """The capacity control: a nonlinear probe on the same mean-pooled vectors.

    The MLP has no exact decomposition the way the linear probe does — its
    logit is not a fixed projection of the pooled vector. Its token scores are
    therefore its integrated-gradient direction, which closes exactly by IG
    completeness and then redistributes over tokens by the same pooling
    identity (see the attribution section of src/probe/mlp_probe.py). Like
    `attention_probe`, and unlike `linear_probe`, the resulting term is a
    contribution IN CONTEXT: the direction is refitted per post.
    """
    pv = ctx.post_vectors
    model = mlp_probe.fit(pv, ctx.y, ctx.train_idx, seed=ctx.seed)
    steps = int(ctx.cfg["probe"].get("mlp_ig_steps", mlp_probe.IG_STEPS))
    # Enforce completeness once per fold rather than trusting it, as the
    # attention probe does with its own identity.
    residual = mlp_probe.verify_attribution(
        model, pv[ctx.train_idx[:16]], steps)
    acts, ids = ctx.acts, ctx.all_ids
    return Fitted(
        post_scores=lambda idx: mlp_probe.score(model, pv, idx),
        token_scores=lambda i: mlp_probe.token_contributions(
            model, acts[ids[i]]["vecs"].astype(np.float64), steps),
        token_source=f"mlp_probe integrated-gradient direction ({steps} steps, "
                     f"worst completeness residual {residual:.1e} log-odds)",
        score_vectors=lambda V: float(mlp_probe.logit(
            model, np.asarray(V, dtype=np.float64).mean(0))[0]),
        token_vectors_scores=lambda V: mlp_probe.token_contributions(
            model, np.asarray(V, dtype=np.float64), steps),
    )


def fit_afr_probe(ctx: FoldContext) -> Fitted:
    """`linear_probe` with AFR's reweighted last-layer retrain on top.

    The base probe, the reweighting split and the retrained head all live
    inside this fold's training posts. The composite is affine, so the token
    readout is the same projection applied to a token instead of a post vector
    — the identical decomposition the label probe has, on the reweighted
    direction. See src/probe/afr_probe.py.
    """
    pv, acts, ids = ctx.post_vectors, ctx.acts, ctx.all_ids
    model = afr_probe.fit(pv, ctx.y, ctx.train_idx, ctx.seed, ctx.cfg,
                          C=ctx.cfg["probe"]["logreg_C"])
    a, b = model["a"], model["b"]
    return Fitted(
        post_scores=lambda idx: pv[idx] @ a + b,
        token_scores=lambda i: acts[ids[i]]["vecs"].astype(np.float64) @ a + b,
        token_source="afr_probe projection of the token vector",
        provenance=model["provenance"],
        score_vectors=lambda V: float(np.asarray(V, dtype=np.float64).mean(0) @ a + b),
        token_vectors_scores=lambda V: np.asarray(V, dtype=np.float64) @ a + b,
    )


def fit_last_token_probe(ctx: FoldContext) -> Fitted:
    """L2 logistic regression on each post's LAST cached token vector.

    The decoder-only default read-out: the final post position has attended
    to the whole post, so its state is the usual "sentence vector". Same
    solver and C as `linear_probe`; the only change is which vector is the
    post vector. It has NO additive token decomposition -- the pooling is a
    one-hot, so every other token's share is zero -- and `token_scores` is
    None accordingly. Its word ranking is gradient x input through the frozen
    backbone (src.stage2.gradient_input), which writes its Stage 2 cells.
    `score_vectors` scores an arbitrary token matrix by its last row, which is
    what the word-deletion audit needs to re-score a shortened post.
    """
    last, a, b = last_token_affine(ctx)
    return Fitted(
        post_scores=lambda idx: last[idx] @ a + b,
        token_scores=None,
        token_source="last_token_probe: one-hot pooling, no token "
                     "decomposition (ranked by gradient x input instead)",
        score_vectors=lambda V: (
            float(np.asarray(V, dtype=np.float64)[-1] @ a + b)
            if len(V) else float(b)),
        token_vectors_scores=None,
    )


def last_token_affine(ctx: FoldContext):
    """(last-token matrix [N, H], a, b): the fitted read-out as an affine map.

    Exposed on its own because src.stage2.gradient_input applies exactly this
    read-out inside a differentiable forward of the frozen backbone; sharing
    the fit is what makes that attribution the probe's own.
    """
    acts, ids = ctx.acts, ctx.all_ids
    last = np.stack([np.asarray(acts[pid]["vecs"], dtype=np.float64)[-1]
                     for pid in ids])
    probe = linear_probe.fit(last[ctx.train_idx],
                             ctx.y[ctx.train_idx].astype(int),
                             "logistic", ctx.seed,
                             C=ctx.cfg["probe"]["logreg_C"])
    a, b = _affine_of(probe, last[ctx.train_idx][:64])
    return last, a, b


FITTERS = {
    "linear_probe": fit_linear_probe,
    # the same fitter on the cache's final-layer file (common.probe_layer_of)
    "final_linear_probe": fit_linear_probe,
    "final_attention_probe": fit_attention_probe,
    "last_token_probe": fit_last_token_probe,
    "mlp_probe": fit_mlp_probe,
    "attention_probe": fit_attention_probe,
    "shrunk_attention_probe": fit_shrunk_attention_probe,
    "attention_mlp_probe": fit_attention_mlp_probe,
    "afr_probe": fit_afr_probe,
}

# Methods whose published row exists once per variant rather than once per
# method. Empty: every method here publishes exactly one row per (method,
# model). Kept as the hook the CLIs loop over so a method that does split its
# rows has one place to say so.
VARIANTS_OF = {}


def for_fold(ctx: FoldContext, fold: int) -> FoldContext:
    """The same context with `train_idx` set to every post outside `fold`."""
    from dataclasses import replace

    return replace(ctx, train_idx=np.flatnonzero(ctx.fold_col != fold))


# =========================================================== 3. span scoring


# --------------------------------------------------------------- token labels


def token_masks(offsets: np.ndarray, text: str, span_start: int, span_end: int):
    """(content mask, gold mask) over one post's cached tokens.

    "Content token" follows the SAE audit's definition: a token counts only if
    it overlaps a word that survives normalisation, so pure-punctuation tokens
    are neither scored nor counted in the base rate. Gold membership is
    character overlap with the psychiatrist span.
    """
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    tok_s, tok_e = offsets[:, 0], offsets[:, 1]
    kept = [w for w in words(text) if w["norm"]]
    if not kept:
        return np.zeros(len(tok_s), dtype=bool), np.zeros(len(tok_s), dtype=bool)
    w_s = np.array([w["start"] for w in kept], dtype=np.int64)
    w_e = np.array([w["end"] for w in kept], dtype=np.int64)
    content = ((tok_s[:, None] < w_e[None, :]) & (tok_e[:, None] > w_s[None, :])).any(1)
    gold = (tok_s < int(span_end)) & (tok_e > int(span_start)) & content
    return content, gold


def post_token_auprc(scores: np.ndarray, content: np.ndarray,
                     gold: np.ndarray) -> float:
    """AUPRC over one post's content tokens; NaN where it is undefined."""
    s = np.asarray(scores, dtype=np.float64)[content]
    g = gold[content]
    if len(s) == 0 or not np.isfinite(s).all() or len(np.unique(g)) < 2:
        return float("nan")
    return float(average_precision_score(g.astype(int), s))


# ---------------------------------------------------------------- word labels


@dataclass(frozen=True)
class WordTargets:
    """One post's word-level scoring structure, fixed by the corpus.

    `alignment` is a `sae_features.TokenAlignment` carrying one (token, word)
    pair per character-span overlap — a token straddling two words is offered
    to both — which `sae_features.aggregate_word_scores` reduces per word.
    `scored` holds the indices into `words(text)` of the words at least one
    cached token reaches, and `gold` marks which of *those* overlap the
    psychiatrist span.
    """

    alignment: object
    scored: np.ndarray
    gold: np.ndarray


def word_targets(offsets: np.ndarray, text: str,
                 span_start: int, span_end: int) -> WordTargets:
    """The word-level counterpart of `token_masks`, on the same overlap rule."""
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    tok_s, tok_e = offsets[:, 0], offsets[:, 1]
    all_words = words(text)
    kept = np.array([i for i, w in enumerate(all_words) if w["norm"]], dtype=np.int64)
    w_s = np.array([all_words[i]["start"] for i in kept], dtype=np.int64)
    w_e = np.array([all_words[i]["end"] for i in kept], dtype=np.int64)
    overlap = ((tok_s[:, None] < w_e[None, :]) & (tok_e[:, None] > w_s[None, :]))
    tok_pos, word_pos = np.nonzero(overlap)
    gold_token = (tok_s < int(span_end)) & (tok_e > int(span_start))
    alignment = _sae_profile().TokenAlignment(
        token_indices=tok_pos.astype(np.int64),
        word_indices=kept[word_pos],
        gold_mask=gold_token[tok_pos],
        n_words=len(all_words),
        truncated=False,
    )
    reached = overlap.any(axis=0)
    gold_word = (w_s < int(span_end)) & (w_e > int(span_start))
    return WordTargets(alignment, kept[reached], gold_word[reached])


def post_word_auprc(scores: np.ndarray, targets: WordTargets,
                    reduction: str) -> float:
    """AUPRC over one post's scored words; NaN where it is undefined.

    Mirrors `post_token_auprc` decision for decision: a post is NaN when a
    scored word's value is non-finite, or when its words are single-class.
    `scores` is one score per cached token.
    """
    if len(targets.scored) == 0:
        return float("nan")
    word_scores = _sae_profile().aggregate_word_scores(
        np.asarray(scores, dtype=np.float64), targets.alignment,
        reduction=reduction)
    s = word_scores[targets.scored]
    g = targets.gold
    if not np.isfinite(s).all() or len(np.unique(g)) < 2:
        return float("nan")
    return float(average_precision_score(g.astype(int), s))


# --------------------------------------------------------------- per-post prep


def span_structures(posts, acts_offsets_of, plaus_ids):
    """(masks, targets, base rates, word counts) for a plausibility subset.

    `acts_offsets_of(pid)` returns the cached token offsets for one post.
    Base rates: a post whose content tokens (or scored words) are all gold, or
    none of them, has no ranking metric — chance is quoted over the rest.
    """
    by_id = posts.set_index("id")
    pos = {pid: i for i, pid in enumerate(plaus_ids)}
    masks, targets = {}, {}
    token_base = np.full(len(plaus_ids), np.nan)
    word_base = np.full(len(plaus_ids), np.nan)
    word_counts = np.zeros(len(plaus_ids), dtype=np.int64)
    for pid in plaus_ids:
        row = by_id.loc[pid]
        span = (int(row["span_start"]), int(row["span_end"]))
        offsets = acts_offsets_of(pid)
        content, gold = token_masks(offsets, row["text"], *span)
        masks[pid] = (content, gold)
        targets[pid] = word_targets(offsets, row["text"], *span)
        n_content, n_gold = int(content.sum()), int(gold.sum())
        if 0 < n_gold < n_content:
            token_base[pos[pid]] = n_gold / n_content
        n_words = len(targets[pid].scored)
        n_gold_words = int(targets[pid].gold.sum())
        word_counts[pos[pid]] = n_words
        if 0 < n_gold_words < n_words:
            word_base[pos[pid]] = n_gold_words / n_words
    return masks, targets, token_base, word_base, word_counts
