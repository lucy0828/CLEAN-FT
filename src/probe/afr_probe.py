"""AFR: automatic feature reweighting of the readout, on frozen features.

Not run directly — src/stage1/train.py, src/stage2/dla_auprc.py and
src/utils/faithfulness.py drive it.

WHAT THIS IS
------------
Automatic Feature Reweighting (Qiu, Potapczynski, Izmailov & Wilson, ICML
2023; github.com/AndPotap/afr). AFR retrains only the last layer of an
already-trained base model on a held-out split, weighting each example by how
badly the base model predicted it:

    w_i = exp(-gamma * p_base(y_i | x_i))

Examples the base model gets wrong are disproportionately the ones whose label
disagrees with whatever shortcut the base model latched onto, so upweighting
them recovers minority performance WITHOUT ever reading a group annotation.

That last clause is why it is in this study. Every other way of resisting a
shortcut here needs the shortcut named in advance: a product-of-experts arm
needs its lexicon pinned up front, an OOD split needs the shift chosen by hand.
AFR needs neither. It is the baseline that asks whether a readout can find the
minority pattern on its own — which is exactly the claim a reader will press
on, so it is a baseline this study has to carry rather than one it may skip.

THE PORT, AND THE ONE DEVIATION
-------------------------------
AFR's base model is normally an ERM-*fine-tuned* network. Here the base is
`linear_probe` — the same L2 logistic regression at the same fixed C, on
the same mean-pooled vectors at the same 0.65-depth layer — so the whole
method stays inside this study's frozen-backbone contract and the only thing
separating `afr_probe` from `linear_probe` is the reweighting.

This is a smaller deviation than it looks. AFR's second stage already runs on
precomputed embeddings (`train_embeddings.py` in the reference repo); only the
provenance of those embeddings changes, from a fine-tuned trunk to a frozen
one. It is still a deviation, and it is the honest reading of any AFR row
here: this measures AFR's reweighting on frozen features, not AFR as published
on top of a fine-tuned trunk. The fine-tuned-base version is a separate arm and
is not what this module is.

Binary, not K-way: `losses.get_exp_weights` reads a softmax probability of the
true class. Under the one-logit sigmoid parameterization used throughout this
repo that probability is sigmoid(z) for a `present` post and 1 - sigmoid(z)
for an `absent` one — the same number a 2-class softmax returns, with the
redundant direction dropped. Keeping one logit is also what leaves the readout
affine, which is what gives Stage 2 its token decomposition for free.

THE SPLIT, AND NO LEAKAGE
-------------------------
AFR needs two disjoint splits: one to fit the base model, one to reweight and
retrain on. Both are carved out of THIS FOLD'S TRAINING POSTS, class-stratified
(`configs/default.yaml`, `afr.holdout_prop`, matching the reference repo's
`--train_prop=-20`). The held fold contributes to neither and is scored only at
the end, exactly as it is for every other probe.

Stratifying is not cosmetic: the class rebalance below divides by a per-class
count, so a reweighting split that happened to miss a class would divide by
zero rather than fail loudly.

WHY THE OPTIMIZER IS COPIED RATHER THAN IMPROVED ON
---------------------------------------------------
The second-stage objective is convex, so it is tempting to solve it exactly
with L-BFGS instead of replaying AFR's Adam loop. That would be wrong at the
published hyperparameters. With `reg_coeff = 0` the weighted logistic
objective on a separable split is unbounded — the exact optimizer would run the
weights off to infinity. AFR's finite step count from the ERM warm start is
doing real regularization work, and dropping it would not be a cleaner version
of the method, it would be a different and broken one. So the loop below is
AFR's: warm start at the base head, Adam at a fixed learning rate, a fixed
number of full-batch steps.

WHAT IS PINNED, AND THE LIMITATION THAT COMES WITH IT
------------------------------------------------------
AFR selects `gamma` per dataset on a validation set that HAS group labels.
This study has no group labels, so there is nothing here to select against,
and inventing a selection criterion would quietly spend an annotation budget
the study claims not to have. `gamma` is therefore pinned to one of AFR's OWN
published values and never searched on this corpus.

Which one is not arbitrary. AFR publishes 10.0 for Waterbirds and 1.44 for
CelebA, and gamma is only meaningful relative to how confident the base model
is: the weight exp(-gamma * p_true) spans exp(gamma) between a p_true of 0 and
one of 1. Waterbirds' ERM base is near-certain on most of its held-out split,
so gamma=10 still leaves a usable spread. The base probe here is not: mean
p_true is 0.72 (gemma-2-2b layer 17, fold 0), and at gamma=10 the effective
sample size of the reweighting split falls to 131 of 634 posts and held-fold
AUROC drops to 0.772, BELOW the 0.836 of the unweighted probe it is supposed
to improve. At 1.44 the effective sample is 422 and AUROC is 0.855.

Measured on gemma-2-2b layer 17, fold 0 (held-fold AUROC against the 0.836 of
`linear_probe`, and the effective sample size of the 634-post reweighting
split):

    gamma    0.5     1.44    3.0     5.0     10.0
    AUROC    0.857   0.855   0.853   0.847   0.772
    ESS      598     422     221     158     131

So the pin is 1.44, and the ladder above is the sensitivity a reader should be
shown rather than one flattering point -- fold 0 is kind to AFR and one fold is
not the result. Pooled over the five study folds the published row is 0.830
against `linear_probe`'s 0.837 (results/detection.csv), i.e. the
reweighting does NOT buy in-distribution AUROC here.

That is the expected shape, not a failure: AFR trades average accuracy for
minority accuracy, and a table whose only axis is pooled AUROC has nowhere to
show the side of the trade AFR is for. An AFR row is worth reading against a
group, OOD or counterfactual split; against pooled AUROC alone it is close to
uninformative, and saying so here is cheaper than having a reader discover it.

State the rest as a limitation: an AFR row here is AFR at a
transplanted hyperparameter, and a gamma tuned against group labels could do
better. What it cannot do is quietly do better because it was tuned while the
other rows were not.

THE AFFINE FORM, AND WHY THE WORD EVIDENCE STILL EXISTS
--------------------------------------------------------
The retrained head is linear on the standardized pooled vector, and the
standardizer is affine, so the composite flattens the same way
`linear_probe` does:

    score(x) = x . a + b,   a = W / scale,   b = b_head - (mean / scale) . W

Two consequences the rest of the study depends on, identical to the label
probe's:

* an affine map commutes with the mean, so the pooling identity holds exactly
  and a post's score is the token-count-weighted mean of its token scores;
* the same `a` that scores a post scores each of its words, so Stage 2's
  own-decomposition word AUPRC needs no extra machinery here either.

`fit` asserts the flattened form reproduces the head it came from, so a change
to the standardizer cannot silently desynchronise the two.
"""

import numpy as np

from src.probe import linear_probe
from src.utils.common import best_f1_threshold, unit


def split_holdout(train_idx: np.ndarray, y: np.ndarray, prop: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(base_idx, reweight_idx): AFR's two splits, class-stratified.

    Both come out of `train_idx`, so the held fold contributes to neither.
    `prop` is the reweighting split's share (the reference repo's
    `--train_prop=-20` is `prop=0.2`).
    """
    train_idx = np.asarray(train_idx)
    if not 0.0 < prop < 1.0:
        raise ValueError(f"afr.holdout_prop must be in (0, 1), got {prop}")
    rng = np.random.default_rng(seed)
    labels = np.asarray(y)[train_idx].astype(int)
    base, reweight = [], []
    for cls in (0, 1):
        members = train_idx[labels == cls]
        if len(members) < 2:
            raise ValueError(
                f"class {cls} has {len(members)} training posts; AFR needs at "
                "least one on each side of its split")
        members = members[rng.permutation(len(members))]
        # Clamped so neither side of the split can come out empty.
        cut = min(max(int(round(len(members) * (1.0 - prop))), 1),
                  len(members) - 1)
        base.append(members[:cut])
        reweight.append(members[cut:])
    return np.sort(np.concatenate(base)), np.sort(np.concatenate(reweight))


def afr_weights(logits: np.ndarray, y: np.ndarray, gamma: float,
                rebalance: bool) -> tuple[np.ndarray, np.ndarray]:
    """(w, p_true): AFR's example weights, in the reference repo's order.

    `losses.get_exp_weights`     w = exp(-gamma * p_true)
    `utils.normalize_weights`    w = w / w.sum()
    `utils.rebalance_weights`    w[y == c] *= len(y) / count(c)   (optional)

    The rebalance deliberately undoes the sum-to-one normalization — it
    rescales the loss as well as reweighting it. That is what the reference
    implementation does, and it is reproduced rather than tidied up.
    """
    p_pos = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    labels = np.asarray(y).astype(int)
    p_true = np.where(labels == 1, p_pos, 1.0 - p_pos)
    w = np.exp(-float(gamma) * p_true)
    w = w / w.sum()
    if rebalance:
        for cls in np.unique(labels):
            members = labels == cls
            w[members] *= len(labels) / members.sum()
    return w, p_true


def _retrain_head(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                  w0: np.ndarray, b0: float, acfg: dict, seed: int):
    """AFR's second stage: `take_step` in the reference repo, full batch.

    loss = sum_i w_i * BCE(z_i, y_i) + reg_coeff * (||W - W0||^2 + (b - b0)^2)

    warm-started at the base head `(w0, b0)` and run for a fixed number of
    steps — see the module docstring on why the step count is load-bearing.
    """
    import torch

    torch.manual_seed(seed)
    X = torch.tensor(Z, dtype=torch.float64)
    target = torch.tensor(np.asarray(y).astype(np.float64), dtype=torch.float64)
    weights = torch.tensor(np.asarray(w, dtype=np.float64), dtype=torch.float64)
    init_w = torch.tensor(np.asarray(w0, dtype=np.float64), dtype=torch.float64)
    init_b = torch.tensor(float(b0), dtype=torch.float64)

    weight = init_w.clone().requires_grad_(True)
    bias = init_b.clone().requires_grad_(True)
    opt = torch.optim.Adam([weight, bias], lr=float(acfg["lr"]))
    reg_coeff = float(acfg["reg_coeff"])
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(int(acfg["steps"])):
        opt.zero_grad()
        logits = X @ weight + bias
        loss = (weights * lossf(logits, target)).sum()
        if reg_coeff:
            loss = loss + reg_coeff * (
                (weight - init_w).pow(2).sum() + (bias - init_b).pow(2)
            )
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = float((weights * lossf(X @ weight + bias, target)).sum())
    return (weight.detach().numpy(), float(bias.detach()), final)


def fit(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int,
        cfg: dict, C: float = 1.0) -> dict:
    """AFR on frozen mean-pooled vectors, fitted on the training folds only.

    `X` is every post's mean-pooled vector and `y` its label; `train_idx`
    selects the training folds. Returns the same shape of dict
    `linear_probe.fit` returns, plus the composite affine form `(a, b)`
    and the provenance the raw cells carry.
    """
    acfg = cfg["afr"]
    base_idx, reweight_idx = split_holdout(
        train_idx, y, float(acfg["holdout_prop"]), seed)

    # Stage 1: the base model is exactly the label probe, on the base split.
    base = linear_probe.fit(X[base_idx], np.asarray(y)[base_idx].astype(int),
                                  "logistic", seed, C=C)
    scaler = base["scaler"]
    # The base split's standardizer is used everywhere after this. It never saw
    # the reweighting split or the held fold, so neither leaks into the fit.
    Z = scaler.transform(X)

    # Stage 2: weight the reweighting split by how badly the base model does on
    # it, then retrain the head there and nowhere else.
    base_logits = base["clf"].decision_function(Z[reweight_idx])
    y_re = np.asarray(y)[reweight_idx].astype(int)
    w, p_true = afr_weights(base_logits, y_re, acfg["gamma"],
                            bool(acfg["rebalance_weights"]))
    head_w, head_b, final_loss = _retrain_head(
        Z[reweight_idx], y_re, w, base["clf"].coef_[0],
        float(base["clf"].intercept_[0]), acfg, seed)

    # Flatten the standardizer into the head so one dot product scores a post
    # (and, unchanged, a word).
    scale = np.maximum(scaler.scale_, 1e-12)
    a = head_w / scale
    b = float(head_b - (scaler.mean_ / scale) @ head_w)
    reference = X[reweight_idx][:64]
    assert np.allclose(reference @ a + b,
                       scaler.transform(reference) @ head_w + head_b,
                       atol=1e-6), (
        "flattened AFR readout disagrees with the head it was flattened from"
    )

    proj = X[train_idx] @ a + b
    return {
        "kind": "afr",
        "a": a,
        "b": b,
        "w": unit(a),
        "scaler": scaler,
        "base": base,
        "threshold": best_f1_threshold(proj, np.asarray(y)[train_idx].astype(int)),
        # Provenance: an AFR row is only readable next to how hard the
        # reweighting actually pushed. A weight distribution that is nearly
        # uniform means AFR did nothing and the row is `linear_probe`
        # with extra steps, which is a finding rather than a bug.
        "provenance": {
            "afr_n_base": int(len(base_idx)),
            "afr_n_reweight": int(len(reweight_idx)),
            "afr_gamma": float(acfg["gamma"]),
            "afr_reg_coeff": float(acfg["reg_coeff"]),
            "afr_rebalanced": bool(acfg["rebalance_weights"]),
            "afr_base_acc_on_reweight": float(
                ((base_logits > 0).astype(int) == y_re).mean()),
            "afr_weight_ratio": float(w.max() / w.min()),
            "afr_weight_gini": float(_gini(w)),
            "afr_mean_p_true": float(p_true.mean()),
            "afr_final_loss": float(final_loss),
        },
    }


def _gini(w: np.ndarray) -> float:
    """Gini coefficient of the weight vector: 0 uniform, ->1 concentrated.

    One number for "did the reweighting single anyone out", which the ratio
    alone does not give — a single outlier moves max/min but not the bulk.
    """
    v = np.sort(np.asarray(w, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return float("nan")
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum()) / (n * v.sum()) - (n + 1) / n)


def score(model: dict, vecs: np.ndarray) -> np.ndarray:
    """Project post / word / sentence vectors on the fitted AFR readout.

    NaN-safe in the same way `linear_probe.score` is: words with no
    overlapping token keep NaN and rank last.
    """
    out = np.full(len(vecs), np.nan)
    ok = ~np.isnan(vecs).any(axis=1)
    if ok.any():
        out[ok] = vecs[ok] @ model["a"] + model["b"]
    return out
