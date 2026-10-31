"""Linear probe: a mean-pooled logistic direction fitted from post-level labels alone.

Not run directly — src/stage1/train.py, src/stage2/dla_auprc.py and src/utils/faithfulness.py drive it.

The probe is logistic regression on standardized post vectors; the weight
vector, unit-normalized, is the direction.

Nothing is tuned. The layer is fixed a priori at 65% of depth and C
is fixed at 1.0 (configs/default.yaml), because a probe that selects its layer
on labels is quietly spending part of the annotation budget it claims to save.

Word evidence needs no extra machinery: the same w that scores a post scores
each of its words, since the post vector is the mean of its tokens and the dot
product commutes with pooling.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.common import best_f1_threshold, unit


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int, C: float = 1.0) -> dict:
    """Standardize, then L2 logistic regression at fixed C."""
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    proj = clf.decision_function(scaler.transform(X))
    return {
        "kind": "logistic",
        "scaler": scaler,
        "clf": clf,
        "C": C,
        "w": unit(clf.coef_[0]),
        "threshold": best_f1_threshold(proj, y),
    }


def fit(X: np.ndarray, y: np.ndarray, kind: str, seed: int, C: float = 1.0) -> dict:
    if kind == "logistic":
        return fit_logistic(X, y, seed, C)
    raise ValueError(f"unknown probe kind {kind!r}")


def score(model: dict, vecs: np.ndarray) -> np.ndarray:
    """Project post / word / sentence vectors on the fitted direction. NaN-safe:
    words with no overlapping token keep NaN and rank last."""
    out = np.full(len(vecs), np.nan)
    ok = ~np.isnan(vecs).any(axis=1)
    if not ok.any():
        return out
    out[ok] = model["clf"].decision_function(model["scaler"].transform(vecs[ok]))
    return out
