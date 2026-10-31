"""Attention-pooled MLP probe: the missing cell of the pooling x capacity 2x2.

Not run directly -- src/stage1/train.py drives it.

The study already varies two things independently, and until now only three of
the four combinations existed:

                        linear head          nonlinear head
    mean pooling        linear_probe         mlp_probe
    attention pooling   attention_probe      THIS

`mlp_probe` is the capacity control at fixed (mean) pooling; `attention_probe`
is the pooling control at fixed (linear) capacity. This is the interaction
term. It answers one question and no others: **once the pooling is right, does
adding capacity buy anything?**

If this probe lands on `attention_probe`, capacity is irrelevant given a
learned query, and the study should keep the linear head -- which is the one
with an exact additive decomposition. If it lands above, the two factors
combine and the paper's "pooling, not capacity" claim needs qualifying.

WHAT THIS PROBE GIVES UP
------------------------
`attention_probe` keeps a weakened but EXACT decomposition:

    post score - b = sum_t alpha_t (w . X_t)

An MLP head destroys it outright: the logit is not a projection of the pooled
vector, so no fixed w exists and no rearrangement recovers one. Token scores
here are the same integrated-gradient construction `mlp_probe` uses -- an IG
direction from the train-mean baseline, redistributed over tokens by the
attention weights -- so they close by IG completeness rather than by identity,
and they are contributions IN CONTEXT twice over: the direction is refitted per
post AND alpha is a softmax over the whole post.

That is why this is written as a diagnostic and not as a candidate readout.
Stage 2's own-decomposition audit rests on exactness; a probe that only closes
approximately cannot carry it. Run it to settle the 2x2, report the AUROC, and
do not promote it without deciding what happens to Stage 2.

The head matches `mlp_probe`'s one hidden layer of 256 ReLU units, so the
capacity axis of the 2x2 is the same size in both rows. The query, the
optimiser, the step count and the seed handling all match `attention_probe`,
so the pooling axis is the same in both columns.
"""

from __future__ import annotations

import numpy as np
import torch

from src.utils.common import best_f1_threshold, set_seed

SOLVER_NOTE = ("torch minibatch Adam over ragged posts, softmax pooling and "
               "one hidden layer in-graph")
HIDDEN_UNITS = 256          # matches mlp_probe.fit's default
IG_STEPS = 256              # matches probe.mlp_ig_steps


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


def _pool(X, mask, q, hidden):
    s = (X @ q) / (hidden ** 0.5)
    alpha = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=1)
    return (alpha.unsqueeze(-1) * X).sum(1)


def _head(pooled, W1, b1, w2, b2):
    return torch.relu(pooled @ W1 + b1) @ w2 + b2


def fit(acts: dict, posts_df, train_ids, seed: int, cfg: dict) -> dict:
    """Solve (q, W1, b1, w2, b2) jointly on the frozen tokens.

    Query and head are fitted together, exactly as attention_probe fits query
    and direction together: pooling that is chosen before the readout is a
    different (and weaker) method.
    """
    set_seed(seed)
    ap = cfg["attention_pool"]
    train_ids = list(train_ids)
    label = posts_df.set_index("id")["label"]
    feats = [acts[pid]["vecs"] for pid in train_ids]
    y_np = (label.loc[train_ids] == "present").astype(int).to_numpy()
    hidden = feats[0].shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    units = int(cfg.get("probe", {}).get("attn_mlp_units", HIDDEN_UNITS))

    y = torch.tensor(y_np, dtype=torch.float32, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(hidden, generator=g) * ap["init_scale"]).to(dev).requires_grad_(True)
    # He init on the hidden layer; the output layer starts at zero so the
    # probe begins at the same all-logits-zero point attention_probe does.
    W1 = (torch.randn(hidden, units, generator=g)
          * (2.0 / hidden) ** 0.5).to(dev).requires_grad_(True)
    b1 = torch.zeros(units, device=dev, requires_grad=True)
    w2 = torch.zeros(units, device=dev, requires_grad=True)
    b2 = torch.zeros(1, device=dev, requires_grad=True)
    params = [q, W1, b1, w2, b2]
    opt = torch.optim.Adam(params, lr=ap["lp_lr"])
    lossf = torch.nn.BCEWithLogitsLoss()

    batch = int(ap.get("lp_batch", 128))
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for _ in range(int(ap["lp_steps"])):
        if len(order) < batch:
            order += rng.permutation(len(feats)).tolist()
        idx, order = order[:batch], order[batch:]
        X, mask = _pad_batch([feats[i] for i in idx], dev)
        opt.zero_grad()
        loss = lossf(_head(_pool(X, mask, q, hidden), W1, b1, w2, b2), y[idx])
        loss.backward()
        opt.step()
        del X, mask

    with torch.no_grad():
        scores, pooled_all = [], []
        for i in range(0, len(feats), batch):
            X, mask = _pad_batch(feats[i: i + batch], dev)
            pooled = _pool(X, mask, q, hidden)
            pooled_all.append(pooled.cpu().numpy())
            scores.append(_head(pooled, W1, b1, w2, b2).cpu().numpy())
            del X, mask
        train_scores = np.concatenate(scores)

    model = {
        "kind": "attention_mlp",
        "q": q.detach().cpu().numpy().astype(np.float64),
        "W1": W1.detach().cpu().numpy().astype(np.float64),
        "b1": b1.detach().cpu().numpy().astype(np.float64),
        "w2": w2.detach().cpu().numpy().astype(np.float64),
        "b2": float(b2.detach()),
        "hidden": hidden,
        "units": units,
        # The IG baseline: the mean POOLED train vector, the analogue of
        # mlp_probe's train-mean baseline one level up.
        "baseline": np.concatenate(pooled_all).mean(axis=0).astype(np.float64),
        "solver": SOLVER_NOTE,
        "n_posts": len(feats),
        "train_accuracy": float(((train_scores > 0).astype(int) == y_np).mean()),
        "threshold": best_f1_threshold(train_scores, y_np),
    }
    if dev == "cuda":
        torch.cuda.empty_cache()
    return model


# ------------------------------------------------------------------ scoring


def alpha(model: dict, vecs: np.ndarray) -> np.ndarray:
    s = (vecs @ model["q"]) / (model["hidden"] ** 0.5)
    e = np.exp(s - s.max())
    return e / e.sum()


def pooled_vector(model: dict, act) -> np.ndarray:
    vecs = np.asarray(act["vecs"], dtype=np.float64)
    return alpha(model, vecs) @ vecs


def _forward(model: dict, x: np.ndarray) -> np.ndarray:
    h = np.maximum(x @ model["W1"] + model["b1"], 0.0)
    return h @ model["w2"] + model["b2"]


def post_score(model: dict, act) -> float:
    return float(_forward(model, pooled_vector(model, act)[None, :])[0])


def score_vectors(model: dict, V) -> float:
    return post_score(model, {"vecs": np.asarray(V)})


# -------------------------------------------------------------- attribution


def _grad(model: dict, X: np.ndarray) -> np.ndarray:
    """d logit / d pooled-vector, for a batch of pooled vectors."""
    pre = X @ model["W1"] + model["b1"]
    return (pre > 0).astype(np.float64) * model["w2"] @ model["W1"].T


def ig_direction(model: dict, x: np.ndarray, steps: int = IG_STEPS) -> np.ndarray:
    """Path-averaged gradient from the train-mean pooled baseline to `x`."""
    base = model["baseline"]
    t = (np.arange(steps) + 0.5) / steps
    path = base[None, :] + t[:, None] * (x - base)[None, :]
    return _grad(model, path).mean(axis=0)


def token_contributions(model: dict, act, steps: int = IG_STEPS) -> np.ndarray:
    """alpha_t * (g . X_t): the IG direction redistributed over tokens.

    Closes by IG completeness, not by identity -- see the module docstring.
    """
    vecs = np.asarray(act["vecs"], dtype=np.float64)
    a = alpha(model, vecs)
    g = ig_direction(model, a @ vecs, steps)
    return a * (vecs @ g)


def verify_attribution(model: dict, acts: list, steps: int = IG_STEPS) -> float:
    """Largest relative completeness residual over `acts`; raises past 2%."""
    worst = 0.0
    for act in acts:
        x = pooled_vector(model, act)
        g = ig_direction(model, x, steps)
        closed = float(g @ (x - model["baseline"]))
        actual = post_score(model, act) - float(
            _forward(model, model["baseline"][None, :])[0])
        scale = max(abs(actual), 1e-9)
        worst = max(worst, abs(closed - actual) / scale)
    if worst > 0.02:
        raise AssertionError(
            f"integrated-gradient completeness off by {worst:.1%}; raise "
            f"steps (currently {steps})")
    return worst
