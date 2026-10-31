"""Attention probing: a direction plus a learned query over frozen tokens.

Not run directly — src/stage1/train.py, src/stage2/dla_auprc.py and
src/utils/faithfulness.py drive it.

This is the LP-only endpoint of the v2 design: maximal feature preservation,
since no backbone weight moves at all. It reads the same cached per-token
activations at the same a-priori 0.65-depth layer the linear probe
reads, and differs from it in exactly one respect — how the tokens are
aggregated:

    linear probe        post vector = mean of its token vectors
    attention probe     post vector = sum_t alpha_t X_t,  alpha = softmax(X q)

The argument for the change is that mean pooling imposes uniform
weights on a residual stream that has no privileged basis, so it is a poor
readout of a concept carried by a few tokens; a learned query aggregates the
way the transformer's own attention does.

WHAT DECOMPOSITION SURVIVES, EXACTLY
------------------------------------
The v1 faithfulness argument was that a probe's word scores *are* its decision,
because with mean pooling

    post score = w . mean_t(X_t) + b = mean_t(w . X_t) + b

so the post score is the token-count-weighted mean of word scores and the two
cannot disagree (src/utils/probing.verify_identity).

Attention pooling breaks that identity, and the design accepts the cost. What
replaces it is weaker but still exact:

    post score - b = sum_t alpha_t (w . X_t)

so each token still contributes an additive term and the terms still sum to the
decision. The difference is that a token's contribution alpha_t (w . X_t) is no
longer a property of the token alone — alpha is a softmax over the whole post,
so removing one token rescales every other token's weight. Word scores here are
therefore *contributions in context*, not intrinsic word scores, and the number
a word receives depends on what else the post says.

That is a real weakening of the v1 claim and is reported as one: it is why
faithfulness under attention pooling has to be measured empirically by erasure
(Stage 2B) instead of being true by construction. `verify_decomposition` below
checks the additive identity that does hold, so the weaker claim is at least
enforced rather than asserted.

THE QUERY IS THE ONLY UNREGULARIZED OBJECT IN THE READOUT
---------------------------------------------------------
`w` is penalized in every arm that has a penalty (L2 in `linear_probe`, L1 in
the SAE arms). `q` never was: it is `hidden` free parameters fitted by
unconstrained Adam, and its only brake is a small init and a fixed step count.
That is fine while the training pool is large relative to `hidden` and stops
being fine when it is not:

    basis          d(q)     n/d    pooling's AUROC gain over mean pooling
    gemma raw      2,304    1.38   +0.093
    llama raw      4,096    0.77   +0.070
    gemma SAE     16,384    0.19   +0.019
    llama SAE     32,768    0.097  -0.075

The gain is monotone in n/d across two backbones and two bases, which reads as
one overfitting curve rather than as a fact about dictionaries. `solve` below
therefore takes a query penalty: q stays free and is penalized by
q_l2 * ||q||^2, so as q_l2 -> inf the query goes to zero, the softmax goes
uniform, and the arm becomes `linear_probe`. The penalty interpolates the two
ends of the pooling contrast the study already reports, and q_l2=0 is the
published `attention_probe` exactly.

`q_l2` IS SELECTED ON INNER FOLDS, NEVER ON THE HELD FOLD
----------------------------------------------------------
configs/default.yaml refuses tuning, and it is right to: a probe that picks a
hyper-parameter on the evaluation fold is spending the annotation budget the
study claims to save. `select_q_l2` therefore splits the TRAINING pool by its
own fold column and never touches the outer held fold, so the selection is
nested and the "nothing is tuned on the outcome" claim survives. The selected
value travels on every scored row (`q_l2_selected`) so a reader can see how
hard the query had to be held down on each backbone.
"""

import numpy as np
import torch

from src.utils.common import best_f1_threshold, unit
from src.utils.common import set_seed

SOLVER_NOTE = "torch minibatch Adam over ragged posts, softmax pooling in-graph"


def _pad_batch(feats: list[np.ndarray], device: str):
    """Ragged per-post token matrices -> a padded [B, T, H] batch and its mask."""
    T = max(len(f) for f in feats)
    H = feats[0].shape[1]
    X = torch.zeros(len(feats), T, H, dtype=torch.float32, device=device)
    mask = torch.zeros(len(feats), T, dtype=torch.bool, device=device)
    for i, f in enumerate(feats):
        X[i, : len(f)] = torch.from_numpy(np.ascontiguousarray(f)).float().to(device)
        mask[i, : len(f)] = True
    return X, mask


def _logits(X, mask, q, w, b, hidden):
    s = (X @ q) / (hidden ** 0.5)
    alpha = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=1)
    return ((alpha.unsqueeze(-1) * X).sum(1) @ w) + b


def fit(acts: dict, posts_df, train_ids, seed: int, cfg: dict) -> dict:
    """Solve (q, w, b) on the frozen tokens of the training posts.

    Unlike the mean-pooled probe there is no closed form — alpha depends on q,
    so the objective is not convex in the parameters — but nothing about the
    backbone moves, so this is still a probe: the only thing fitted is how to
    read a representation that is held fixed.

    When `attention_pool.q_l2_select` is set, the query penalty is chosen first
    by `select_q_l2` on folds INSIDE the training pool. `posts_df` carries the
    fold column that split is taken from; the outer held fold is not in
    `train_ids` and so cannot reach the selection.
    """
    train_ids = list(train_ids)
    by_id = posts_df.set_index("id")
    feats = [acts[pid]["vecs"] for pid in train_ids]
    y_np = (by_id["label"].loc[train_ids] == "present").astype(int).to_numpy()
    if cfg["attention_pool"].get("q_l2_select"):
        inner = by_id["fold"].loc[train_ids].to_numpy()
        chosen = select_q_l2(feats, y_np, inner, seed, cfg)
        model = solve(feats, y_np, seed, cfg, q_l2=chosen["q_l2"])
        model["selection"] = chosen
        return model
    return solve(feats, y_np, seed, cfg)


def solve(feats: list, y_np, seed: int, cfg: dict, *,
          q_l2: float | None = None) -> dict:
    """The solver itself: (q, w, b) from ragged token matrices and labels.

    Split out of `fit` because the corpus activation cache is no longer the
    only thing this objective is solved on. `src.baselines.fine_tune`
    runs the same solve on final-layer states pulled from a live backbone, for
    the LP stage of the attention-pooled fine-tune; an LP stage that solved a
    DIFFERENT objective from the frozen attention probe would make the frozen
    and fine-tuned attention arms incomparable. One solver, two callers.

    `feats` is one [T_i, H] array per post and `y_np` its 0/1 labels, aligned
    positionally.

    `q_l2` defaults to the `attention_pool` block and is taken as a keyword only
    by `select_q_l2`, which has to sweep it. The default (`q_l2=0.0`) takes the
    original free-query code path unchanged, penalty term and all — the
    published `attention_probe` rows are reproduced bit for bit, not
    approximately.
    """
    set_seed(seed)
    ap = cfg["attention_pool"]
    q_l2 = float(ap.get("q_l2", 0.0)) if q_l2 is None else float(q_l2)
    y_np = np.asarray(y_np).astype(int)
    hidden = feats[0].shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    y = torch.tensor(y_np, dtype=torch.float32, device=dev)
    w = torch.zeros(hidden, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    q = (torch.randn(hidden, device=dev)
         * ap["init_scale"]).requires_grad_(True)
    params = [q, w, b]
    opt = torch.optim.Adam(params, lr=ap["lp_lr"])
    lossf = torch.nn.BCEWithLogitsLoss()

    # Posts stay on the host and are padded per minibatch: the full training
    # pool densified at once is tens of GB, most of it padding.
    batch = int(ap.get("lp_batch", 128))
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for _ in range(int(ap["lp_steps"])):
        if len(order) < batch:
            order += rng.permutation(len(feats)).tolist()
        idx, order = order[:batch], order[batch:]
        X, mask = _pad_batch([feats[i] for i in idx], dev)
        opt.zero_grad()
        loss = lossf(_logits(X, mask, q, w, b, hidden), y[idx])
        if q_l2:
            # Explicit L2 in the objective, not decoupled weight decay: the
            # penalty is part of the model being fitted, so it has to be part
            # of the loss the fit is reported against. Guarded so that q_l2=0
            # leaves the original graph untouched.
            loss = loss + q_l2 * (q * q).sum()
        loss.backward()
        opt.step()
        del X, mask

    with torch.no_grad():
        scores = []
        for i in range(0, len(feats), batch):
            X, mask = _pad_batch(feats[i : i + batch], dev)
            scores.append(_logits(X, mask, q, w, b, hidden).cpu().numpy())
            del X, mask
        train_scores = np.concatenate(scores)
    model = {
        "kind": "attention",
        "q": q.detach().cpu().numpy().astype(np.float64),
        "w_raw": w.detach().cpu().numpy().astype(np.float64),
        "b": float(b.detach()),
        "w": unit(w.detach().cpu().numpy()),
        "hidden": hidden,
        "solver": SOLVER_NOTE,
        "n_posts": len(feats),
        "train_accuracy": float(((train_scores > 0).astype(int) == y_np).mean()),
        "threshold": best_f1_threshold(train_scores, y_np),
        "q_l2": q_l2,
    }
    if dev == "cuda":
        torch.cuda.empty_cache()
    return model


# ------------------------------------------- selecting the query penalty
#
# The rule, in full, because "selected by cross-validation" is not a
# specification and this study's whole claim is that it does not tune:
#
#   * The candidate grid is fixed in configs/default.yaml, not searched.
#   * The split is the corpus's OWN stratified fold column, restricted to the
#     posts already in the training pool. No new randomness, no new split.
#   * The first `q_l2_inner_folds` fold ids present in the pool are used, in
#     sorted order, so the selection is a deterministic function of (outer
#     fold, seed) and reruns reproduce.
#   * The outer held fold is not in the pool, so it cannot be scored against.
#   * Ties in mean inner AUROC break toward the LARGER penalty. A tie means the
#     data does not distinguish the models, and the more constrained one is the
#     one that was not preferred on evidence.


def score_feats(model: dict, feats: list) -> np.ndarray:
    """Post scores for ragged token matrices — what `select_q_l2` ranks on."""
    return np.array([post_score(model, {"vecs": np.asarray(f)}) for f in feats])


def select_q_l2(feats: list, y_np, inner_folds, seed: int, cfg: dict) -> dict:
    """Pick `q_l2` on folds inside the training pool. Never sees the held fold.

    `inner_folds` is the corpus fold id of each entry of `feats`, so the split
    below reuses the study's own stratified assignment rather than inventing a
    second one.
    """
    from sklearn.metrics import roc_auc_score

    ap = cfg["attention_pool"]
    grid = [float(g) for g in ap["q_l2_grid"]]
    inner_folds = np.asarray(inner_folds)
    y_np = np.asarray(y_np).astype(int)
    n_inner = int(ap.get("q_l2_inner_folds", 3))
    use = sorted(set(inner_folds.tolist()))[:n_inner]

    per_lambda: dict[float, list[float]] = {g: [] for g in grid}
    for f in use:
        tr = np.flatnonzero(inner_folds != f)
        te = np.flatnonzero(inner_folds == f)
        tr_feats = [feats[i] for i in tr]
        te_feats = [feats[i] for i in te]
        for g in grid:
            m = solve(tr_feats, y_np[tr], seed, cfg, q_l2=g)
            per_lambda[g].append(
                float(roc_auc_score(y_np[te], score_feats(m, te_feats))))

    mean = {g: float(np.mean(v)) for g, v in per_lambda.items()}
    best = max(mean.values())
    chosen = max(g for g in grid if mean[g] >= best)   # ties -> larger penalty
    return {
        "q_l2": chosen,
        "grid": grid,
        "inner_folds": [int(f) for f in use],
        "inner_auroc": {str(g): mean[g] for g in grid},
        "rule": "argmax mean inner-fold AUROC, ties to the larger penalty",
    }


def _alpha(model: dict, vecs: np.ndarray) -> np.ndarray:
    s = (vecs @ model["q"]) / (model["hidden"] ** 0.5)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def post_score(model: dict, act: dict) -> float:
    """The attention-pooled decision for one post."""
    vecs = act["vecs"].astype(np.float64)
    return float(_alpha(model, vecs) @ (vecs @ model["w_raw"]) + model["b"])


def token_contributions(model: dict, act: dict) -> np.ndarray:
    """alpha_t * (w . X_t) per token — the additive terms of the decision."""
    vecs = act["vecs"].astype(np.float64)
    return _alpha(model, vecs) * (vecs @ model["w_raw"])


def word_scores(model: dict, act: dict, post_words: list[dict]) -> np.ndarray:
    """A word's contribution: the sum of its tokens' additive terms.

    Summed, not averaged, because these are contributions to one total rather
    than independent scores — the word's share of the decision is what the
    tokens under it jointly put in. Words with no overlapping token keep NaN and
    rank last, matching every other arm's convention.
    """
    contrib = token_contributions(model, act)
    tok_s, tok_e = act["offsets"][:, 0], act["offsets"][:, 1]
    out = np.full(len(post_words), np.nan)
    for wi, w in enumerate(post_words):
        hit = (tok_s < w["end"]) & (tok_e > w["start"])
        if hit.any():
            out[wi] = contrib[hit].sum()
    return out


def attention_entropy(model: dict, act: dict) -> float:
    """H(alpha) / log T — how far this post's pooling is from uniform.

    1.0 is mean pooling, 0.0 is a hard argmax. Reported because the query
    penalty is supposed to move exactly this quantity, and a penalty that
    recovered AUROC by collapsing the arm back to mean pooling would be a
    different result from one that kept a sharp query and made it generalize.
    """
    alpha = _alpha(model, act["vecs"].astype(np.float64))
    t = len(alpha)
    if t < 2:
        return 1.0
    nz = alpha[alpha > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(t))


def verify_decomposition(model: dict, act: dict, atol: float = 1e-6) -> float:
    """Check sum_t alpha_t (w . X_t) + b equals the post score.

    The weaker identity that survives attention pooling (see the module
    docstring). Enforced rather than asserted, so a future change to the pooling
    cannot quietly invalidate the word-level story the arm reports.
    """
    total = float(token_contributions(model, act).sum() + model["b"])
    got = post_score(model, act)
    assert abs(total - got) < max(atol, 1e-9 * abs(got)), (
        f"attention decomposition broken: contributions {total} vs score {got}"
    )
    return abs(total - got)
