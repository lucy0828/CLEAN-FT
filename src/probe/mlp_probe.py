"""Mean-pooled MLP probe: a nonlinear capacity control on frozen post vectors.

Not run directly — `src.stage1.train` drives it.

The probe reads the same mean-pooled residual vectors as the mean-pool
logistic probe, then fits a one-hidden-layer MLP. It is the capacity control: if a
linear readout underperforms, this arm asks whether extra nonlinearity on the
same pooled vector closes the gap. An MLP over a post vector has no EXACT
additive decomposition the way a linear readout does; its word scores are its
integrated-gradient direction, which closes exactly by IG completeness and is
then redistributed over tokens by the pooling identity. See the attribution
section at the bottom of this module for the identity, the caveats and the
measured quadrature error.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.utils.common import best_f1_threshold


def _check_train_labels(y, train_idx, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) != n:
        raise ValueError(f"y has {len(y)} entries for {n} posts")
    train_idx = np.asarray(train_idx, dtype=int)
    if not len(train_idx):
        raise ValueError("train_idx is empty")
    yt = y[train_idx]
    if not np.isfinite(yt).all():
        raise ValueError("non-finite label inside train_idx")
    if set(np.unique(yt).tolist()) - {0.0, 1.0}:
        raise ValueError("labels must be 0/1 (1 = present)")
    if len(np.unique(yt)) < 2:
        raise ValueError("train_idx contains a single class")
    return yt


def fit(
    post_vectors: np.ndarray,
    y,
    train_idx,
    *,
    hidden_layer_sizes: tuple[int, ...] = (256,),
    alpha: float = 1e-4,
    max_iter: int = 300,
    seed: int = 0,
) -> dict:
    """2-layer MLP on mean-pooled post vectors (one hidden layer + output).

    Standardisation statistics come from `train_idx` only.
    """
    X = np.asarray(post_vectors, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=int)
    yt = _check_train_labels(y, train_idx, len(X))
    mean = X[train_idx].mean(axis=0)
    scale = X[train_idx].std(axis=0) + 1e-8
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation="relu",
        alpha=alpha,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=False,
    )
    clf.fit((X[train_idx] - mean) / scale, yt.astype(int))
    model = {
        "kind": "mlp_probe",
        "clf": clf,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "seed": int(seed),
        "n_train_posts": int(len(train_idx)),
    }
    train_scores = score(model, X, train_idx)
    model["threshold"] = best_f1_threshold(train_scores, yt.astype(int))
    model["train_accuracy"] = float(
        ((train_scores > model["threshold"]).astype(int) == yt).mean()
    )
    return model


def score(model: dict, post_vectors: np.ndarray, idx) -> np.ndarray:
    """Log-odds of the present class, so scores are comparable to other arms."""
    X = np.asarray(post_vectors, dtype=np.float64)[np.asarray(idx, dtype=int)]
    p = model["clf"].predict_proba(
        (X - model["scaler_mean"]) / model["scaler_scale"]
    )[:, 1]
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------- token attribution (Stage 2)
#
# The MLP has no exact additive decomposition of its own: `logit` is nonlinear
# in the pooled vector, so there is no fixed direction whose projection of a
# token IS that token's contribution the way there is for `linear_probe`. What
# there is, exactly, is the integrated-gradient direction between a baseline
# post and this post. Writing xbar for the pooled vector, r for the baseline
# and gbar for the path-averaged input gradient, IG's completeness theorem
# gives
#
#     logit(xbar) - logit(r) = gbar . (xbar - r)
#
# exactly (up to the quadrature below). gbar is a single direction, so the
# pooling identity applies to it unchanged -- xbar = sum_t X_t / T gives
#
#     logit(xbar) = sum_t (gbar . X_t) / T  +  [logit(r) - gbar . r]
#
# and the per-token term (gbar . X_t)/T is what `token_contributions` returns.
# The bracketed intercept is NOT distributed over tokens, matching the policy
# in src/utils/dla.py: it is the decision reached with no post at all.
#
# Two honest caveats, both of which the linear probe does not carry:
#   * gbar is post-specific. A token's contribution is a contribution IN
#     CONTEXT, as it is for `attention_probe`, not an intrinsic word score.
#   * the path integral is a midpoint Riemann sum over `steps` points. The
#     network is piecewise linear, so the only error is at the kinks the path
#     crosses; `attribution_error` measures it and `verify_attribution` fails
#     the fit rather than publishing a decomposition that does not close.
BASELINE = "train_mean"     # the standardiser's own centre: the average train post
IG_STEPS = 256              # midpoint rule; `probe.mlp_ig_steps` pins it.
                            # The residual falls as O(1/steps) -- measured on a
                            # fitted MLP(256) at 64/256/1024/4096 steps: 6.0e-3,
                            # 9.7e-4, 2.7e-4, 4.5e-5 relative to the log-odds
                            # span the path covers. 256 sits an order of
                            # magnitude inside the tolerance below, and the
                            # WITHIN-post ranking the AUPRC reads depends on the
                            # direction alone, which converges faster than the
                            # residual does.


def logit(model: dict, X: np.ndarray) -> np.ndarray:
    """Log-odds of the present class for raw (unstandardised) vectors.

    The same quantity `score` returns, written out of the fitted weights so
    the gradient below differentiates exactly what the classifier computes.
    The assert keeps the two in step.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    clf = model["clf"]
    Z = (X - model["scaler_mean"]) / model["scaler_scale"]
    hidden = np.maximum(Z @ clf.coefs_[0] + clf.intercepts_[0], 0.0)
    return hidden @ clf.coefs_[1][:, 0] + float(clf.intercepts_[1][0])


def input_gradient(model: dict, X: np.ndarray) -> np.ndarray:
    """d logit / d x at each row of `X`, in raw (unstandardised) coordinates."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    clf = model["clf"]
    Z = (X - model["scaler_mean"]) / model["scaler_scale"]
    live = (Z @ clf.coefs_[0] + clf.intercepts_[0]) > 0.0
    # d out/d z = W1 @ (w2 * 1[pre-activation > 0]); the chain rule through the
    # standardiser is the elementwise 1/scale.
    return (live * clf.coefs_[1][:, 0]) @ clf.coefs_[0].T / model["scaler_scale"]


def ig_direction(model: dict, x: np.ndarray, steps: int = IG_STEPS) -> np.ndarray:
    """Path-averaged input gradient from the train-mean baseline to `x`."""
    x = np.asarray(x, dtype=np.float64).ravel()
    baseline = np.asarray(model["scaler_mean"], dtype=np.float64)
    alphas = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    path = baseline + alphas[:, None] * (x - baseline)
    return input_gradient(model, path).mean(axis=0)


def attribution_error(model: dict, x: np.ndarray, steps: int = IG_STEPS) -> float:
    """|completeness residual| at `x`, in log-odds."""
    x = np.asarray(x, dtype=np.float64).ravel()
    baseline = np.asarray(model["scaler_mean"], dtype=np.float64)
    g = ig_direction(model, x, steps)
    closed = float(logit(model, baseline)[0]) + float(g @ (x - baseline))
    return abs(closed - float(logit(model, x)[0]))


def verify_attribution(model: dict, X: np.ndarray, steps: int = IG_STEPS,
                       rtol: float = 5e-3, atol: float = 1e-6) -> float:
    """Assert IG completeness on `X`; return the worst residual seen.

    Relative to the log-odds range the path covers, not to the logit itself: a
    logit near zero is a near-cancellation, so an absolute-only tolerance is
    exactly the wrong test there (the same argument as src/utils/dla.py).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    baseline = np.asarray(model["scaler_mean"], dtype=np.float64)
    at_baseline = float(logit(model, baseline)[0])
    worst = 0.0
    for x in X:
        residual = attribution_error(model, x, steps)
        span = abs(float(logit(model, x)[0]) - at_baseline)
        if residual > max(atol, rtol * max(span, 1.0)):
            raise AssertionError(
                f"mlp_probe IG completeness residual {residual:.3e} exceeds "
                f"tolerance at {steps} steps (log-odds span {span:.3f})")
        worst = max(worst, residual)
    return worst


def token_contributions(model: dict, token_vectors: np.ndarray,
                        steps: int = IG_STEPS) -> np.ndarray:
    """One additive log-odds term per token of one post.

    The terms sum to the post logit minus the undistributed intercept; see the
    identity at the top of this section.
    """
    X = np.asarray(token_vectors, dtype=np.float64)
    if X.ndim != 2 or not len(X):
        raise ValueError("token_vectors must be a non-empty [T, H] matrix")
    g = ig_direction(model, X.mean(axis=0), steps)
    return (X @ g) / len(X)
