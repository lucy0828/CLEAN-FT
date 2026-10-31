"""SAE attention probing: a learned query over frozen dictionary latents.

Not run directly -- `src.stage1.train` drives it.

WHAT THIS METHOD IS
-------------------
`sae_probe` swaps the BASIS (raw residual -> pinned SAE latents) and keeps mean
pooling and a linear head. `attention_probe` swaps the POOLING (mean ->
learned query) and keeps the raw basis. This module is the cell where both
change, and it exists because the 2x2 in src/probe/PROBES.md measured pooling
at ~+0.07 to +0.09 AUROC on the raw basis while the dictionary itself was
worth ~+0.03 -- so the pooling was the larger untested gain on the SAE side.

    1. project every post token onto the pinned dictionary, as sae_probe does,
       giving a 16,384-d non-negative sparse latent vector per token;
    2. pool tokens with a LEARNED query in that same latent space,
       alpha = softmax(Z q / sqrt(d_sae));
    3. read out with a sparse linear head, logit = w . sum_t alpha_t z_t + b.

THE QUERY LIVES IN THE SAE BASIS, ON PURPOSE
--------------------------------------------
The alternative -- score tokens with a query over the RAW residual and pool the
latents with those weights -- would be easier (the query would be 2,304-d, not
16,384-d) and worse. It would make the method depend on the residual stream in
two different ways at once, and the attention weights would no longer be a
function of dictionary features, which is the property Stage 2 would want to
audit. Here q and w are both vectors over the same 16,384 named features, so
"which features decided where to look" is as answerable as "which features
decided the label".

THE ADDITIVE DECOMPOSITION SURVIVES
-----------------------------------
Exactly as in attention_probe, and unlike attention_mlp_probe:

    logit - b = w . sum_t alpha_t z_t = sum_t alpha_t (w . z_t)

so per-token contributions sum to the decision with NO residual and no
approximation. The identity is weakened in the same way attention pooling
always weakens it -- alpha is a softmax over the whole post, so a token's
contribution alpha_t (w . z_t) depends on what else the post says, and these
are contributions IN CONTEXT rather than intrinsic token scores. It is not
weakened any further than that. `verify_decomposition` checks it on real data
rather than trusting the algebra.

That is the reason this arm is a candidate readout and `attention_mlp_probe`
is only a diagnostic: an exact decomposition is what Stage 2's own-decomposition
audit rests on.

THE 0.515 IS A DENSE READOUT OF A SPARSE BASIS
----------------------------------------------
`sae_probe` already treats "sparse" as a claim about the fitted coefficients,
not about the penalty, and writes `n_nonzero_coef` on every row so a 14,000-
live L1 head cannot pass as a sparse one. The same count is on this arm. The
gemma-2-2b row being scored (`results/dla_auprc.csv`: word AUPRC 0.515 against
`sae_probe`'s 0.434) was fit with **15,625 of 16,384 coefficients live**. That
is a dense linear head on a sparse dictionary — the one thing an SAE arm
exists not to be.

The cause is the optimiser, not a too-small penalty. Adding `l1 * |w|.sum()`
to the loss and letting Adam descend it is a SUBGRADIENT method, not lasso.
At w_j = 0 the subgradient of |w_j| is any value in [-1, +1]; Adam picks one,
scales it by l1, and the next update leaves zero. After 500 steps every
coordinate that ever felt a gradient is some tiny non-zero number, so the
live count is "features that fired at least once" rather than "features L1
selected". The ~759 exact zeros are coordinates the gradient never touched,
not a sparse support. liblinear, which `sae_probe` uses, does not have this
problem: coordinate descent actually hits zero and stays there.

Three things follow, and they are why the count matters more than the 0.515.

1. **This number is not a sparse-SAE result.** The SAE arm exists so a *small*
   set of named features can carry the decision. A 15,625-coefficient head
   does not test that. It tests whether 16k dictionary coordinates, used
   densely, beat 2,304 residual coordinates. The dictionary is doing
   coordinate work; the sparsity of the *readout* is not.

2. **The comparison to `sae_probe` is not a pooling comparison.** That arm is
   genuinely L1-constrained (liblinear). This one, as scored, is effectively
   unconstrained — it gets all 16,384 features, the other does not. Two
   things changed at once: mean pooling became a learned query, AND the head
   gained the extra freedom the other arm pays an L1 cost to refuse. The
   +0.081 therefore confounds pooling with capacity. Some of it could be the
   extra 16k-wide freedom rather than where the query looked. Until the two
   readouts are compared at matched sparsity, the pooling cell on the SAE
   basis is not identified. The same confound sits on the Stage 1 pair
   (0.921 vs 0.870 AUROC).

3. **The plausibility gain can shrink or grow under a real sparse readout.**
   Token contributions here are `alpha_t * (w . z_t)`. A dense `w` spreads
   that inner product across almost every live latent, so a token ranks high
   if *any* of thousands of features fire on it. A sparse `w` concentrates
   the same mass on a few named features. Concentration can sharpen the
   ranking (mass lands on features that actually fire on the gold span) or
   hollow it out (the surviving features fire on the wrong tokens). The
   0.515 does not tell you which, and re-fitting at `sae_probe`'s sparsity
   is what would.

The code now applies the L1 prox after each Adam step, to `w` only:

    w <- sign(w) * max(|w| - lr * l1, 0)

That is what actually produces exact zeros — the same map lasso uses, which
the subgradient loss never did. It is not by itself a sparse model. The
default `l1` of 1e-4 gives threshold `lr * l1 = 1e-7`, which still leaves
~15,625 live: dead features stay dead, everything that fired stays live.
`sweep_l1` exists to pick a penalty against `sae_probe`'s own non-zero count
on the same fold, so the two arms can be compared at comparable readout
sparsity rather than one being handed 16,384 free parameters. The 0.515 /
0.434 pair is the unmatched comparison and should be read as such until that
sweep is the published row.

L1 applies to `w` only: the query decides where to look, and a sparse query
would be a different (and unmotivated) claim.

THE SOFTMAX TEMPERATURE IS LEARNED, NOT 1/sqrt(d_sae)
-----------------------------------------------------
`attention_probe` divides its logits by sqrt(hidden), the scaled-dot-product
convention. That constant exists because a dot product over `d` DENSE
dimensions accumulates ~d terms, so its variance grows with d. **SAE latents
are not dense**, and copying the convention here was a bug: z_t . q has only
as many terms as the token has ACTIVE latents, so the variance grows with
nnz_t and dividing by sqrt(d_sae) over-shrinks the logits by sqrt(d_sae/nnz).

Measured, that is 13x on gemma (96.9 active of 16,384) and 34x on llama (28.2
of 32,768) -- Llama Scope being both wider and sparser per token makes the two
errors compound. The consequence was not subtle: the median per-post standard
deviation of the softmax logits was 0.80 on gemma and 0.079 on llama, so
llama's alpha was uniform to three decimal places, the learned query expressed
no preference at all, and the arm silently degenerated into a badly-fitted
mean-pooled probe (AUROC 0.847 against sae_probe's 0.898; word AUPRC 0.250
against 0.410). Gemma survived only because its two numbers happened to nearly
cancel.

The fix is to stop guessing the scale from d_sae and let the model set it: a
single learned log-temperature, initialised so the logit spread starts at ~1
on whatever basis the arm is handed. It cannot fail differently across
backbones because it is calibrated per fit rather than derived from a
dictionary width.

Note what this does NOT touch: the decomposition. logit - b = sum_t alpha_t
(w . z_t) holds for ANY alpha, however it was produced, so the arm stays
exactly decomposable and Stage-2-eligible.

A second, separate scaling question is left alone deliberately: without
per-token normalisation a denser token gets a larger raw score simply for
having more latents active, so attention may partly track token density rather
than relevance. Dividing by sqrt(nnz_t) would address that. It is a different
bug from this one and is not fixed here -- one change, one measurement.

FITTING
-------
No closed form -- alpha depends on q -- so this is minibatch Adam like
attention_probe, with the same step count and learning rate so the pooling
axis is the same size in both bases. Latents are held as one scipy CSR over
all corpus tokens (~58M non-zeros, ~460 MB) and the two quantities the model
needs per token, z.q and z.w, are sparse mat-vecs; gradients come back through
a custom autograd Function so nothing densifies a 16,384-d row.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from src.utils.common import best_f1_threshold, set_seed

SOLVER_NOTE = ("torch minibatch Adam over CSR token latents, softmax pooling "
               "in the SAE basis")


class _SparseMatVec(torch.autograd.Function):
    """y = Z @ v for scipy-CSR Z and dense v, with the gradient in v.

    torch's own sparse autograd is uneven across layouts and versions, and the
    matrices here are large and fixed, so the two lines of chain rule are
    written out: forward is a mat-vec, backward is a mat-vec by the transpose.
    """

    @staticmethod
    def forward(ctx, v, Z, Zt):
        ctx.Zt = Zt
        return torch.from_numpy(Z @ v.detach().cpu().numpy().astype(np.float32))

    @staticmethod
    def backward(ctx, g):
        return (torch.from_numpy(ctx.Zt @ g.detach().cpu().numpy()
                                 .astype(np.float32)), None, None)


class _SparseMatVecCuda(torch.autograd.Function):
    """The same two lines of chain rule as `_SparseMatVec`, on device.

    Wrapping the mat-vec in an explicit Function rather than leaning on
    autograd through a sparse layout is deliberate: only the FORWARD op has to
    exist for the chosen layout, so this does not depend on which sparse
    layouts a given torch version happens to differentiate. `Z` and `Zt` are
    torch sparse CSR tensors already on the device.
    """

    @staticmethod
    def forward(ctx, v, Z, Zt):
        ctx.Zt = Zt
        return torch.sparse.mm(Z, v.detach().unsqueeze(1)).squeeze(1)

    @staticmethod
    def backward(ctx, g):
        return (torch.sparse.mm(ctx.Zt, g.detach().unsqueeze(1)).squeeze(1),
                None, None)


def _to_cuda_csr(Z, device):
    Z = Z.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(Z.indptr.astype(np.int64)).to(device),
        torch.from_numpy(Z.indices.astype(np.int64)).to(device),
        torch.from_numpy(Z.data.astype(np.float32)).to(device),
        size=Z.shape, device=device)


def _pad_index(bounds, device):
    """([B, Tmax] gather index, [B, Tmax] mask) for ragged per-post token blocks.

    The CPU path softmaxes one post at a time in a Python loop, which is
    thousands of tiny kernel launches on a GPU and would spend the whole run in
    launch overhead. Posts here top out at 384 tokens, so padding the ragged
    blocks into one [B, Tmax] matrix and softmaxing along a masked axis is the
    same arithmetic in one launch. Padded slots point at row 0 and are masked
    to -inf before the softmax, so they take exactly zero weight.
    """
    lens = np.array([hi - lo for lo, hi in bounds], dtype=np.int64)
    tmax = int(lens.max())
    idx = np.zeros((len(bounds), tmax), dtype=np.int64)
    mask = np.zeros((len(bounds), tmax), dtype=bool)
    for i, (lo, hi) in enumerate(bounds):
        idx[i, : hi - lo] = np.arange(lo, hi)
        mask[i, : hi - lo] = True
    return (torch.from_numpy(idx).to(device),
            torch.from_numpy(mask).to(device))


def _pooled_logits_cuda(Z, Zt, idx, mask, q, w, b, log_temp):
    """`_pooled_logits` on device, over the padded index from `_pad_index`."""
    s = _SparseMatVecCuda.apply(q, Z, Zt) * torch.exp(log_temp)
    v = _SparseMatVecCuda.apply(w, Z, Zt)
    sg = s[idx].masked_fill(~mask, float("-inf"))
    a = torch.softmax(sg, dim=1)
    return (a * v[idx] * mask).sum(dim=1) + b


def stack_token_latents(token_latents, post_ids):
    """(one CSR over every token, [n_posts, 2] row bounds) in `post_ids` order."""
    blocks, bounds, at = [], np.zeros((len(post_ids), 2), dtype=np.int64), 0
    for i, pid in enumerate(post_ids):
        rows = token_latents.rows[pid]
        blocks.append(rows)
        bounds[i] = (at, at + rows.shape[0])
        at += rows.shape[0]
    return sparse.vstack(blocks, format="csr"), bounds


def _pooled_logits(Z, Zt, bounds, q, w, b, log_temp):
    """Post logits for the posts covered by `bounds`.

    `log_temp` is the learned softmax temperature. It replaces the
    1/sqrt(d_sae) of a dense attention layer, which is the wrong constant for a
    sparse basis -- see the module docstring.
    """
    s = _SparseMatVec.apply(q, Z, Zt) * torch.exp(log_temp)
    v = _SparseMatVec.apply(w, Z, Zt)
    out = []
    for lo, hi in bounds:
        a = torch.softmax(s[lo:hi], dim=0)
        out.append((a * v[lo:hi]).sum())
    return torch.stack(out) + b


def sweep_l1(token_latents, posts_df, train_ids, seed, cfg, grid,
             device: str | None = None):
    """(l1, n_nonzero_coef) for each penalty in `grid`, on one fit each.

    Run this against `sae_probe`'s own non-zero count on the same fold to pick
    a penalty, rather than choosing one and reporting whatever sparsity falls
    out.
    """
    out = []
    for l1 in grid:
        m = fit(token_latents, posts_df, train_ids, seed, cfg, l1=l1,
                device=device)
        out.append((l1, m["n_nonzero_coef"], m["train_accuracy"]))
    return out


def fit(token_latents, posts_df, train_ids, seed: int, cfg: dict,
        l1: float | None = None, device: str | None = None,
        q_l2: float | None = None) -> dict:
    """Solve (q, w, b) on the frozen latents of the training posts.

    `device` selects where the two sparse mat-vecs run. None (the default) is
    the scipy/CPU path every published row was fitted with and is left exactly
    as it was. A cuda device runs the identical arithmetic on device -- same
    seed, same 500 Adam steps, same prox -- and exists because an l1 sweep is
    one fit per grid point and the CPU path is minutes each.
    """
    set_seed(seed)
    ap = cfg["attention_pool"]
    if l1 is None:
        l1 = float(cfg.get("sae_attention", {}).get("l1", 1e-4))
    if q_l2 is None:
        q_l2 = float(cfg.get("sae_attention", {}).get("q_l2", 0.0))
    train_ids = list(train_ids)
    label = posts_df.set_index("id")["label"]
    y_np = (label.loc[train_ids] == "present").astype(int).to_numpy()
    Z, bounds = stack_token_latents(token_latents, train_ids)
    Z = Z.astype(np.float32)
    Zt = Z.T.tocsr()
    d_sae = Z.shape[1]

    dev = torch.device(device) if device else None
    if dev is not None and dev.type == "cuda":
        Z_dev, Zt_dev = _to_cuda_csr(Z, dev), _to_cuda_csr(Zt, dev)
        idx_all, mask_all = _pad_index(bounds, dev)

    y = torch.tensor(y_np, dtype=torch.float32, device=dev)
    # The generator stays on CPU and the draw is moved: a cuda generator with
    # the same seed does NOT reproduce the CPU draw, and the point of the
    # device path is to change where the arithmetic runs, not what is fitted.
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(d_sae, generator=g) * ap["init_scale"]).to(dev).requires_grad_(True)
    w = torch.zeros(d_sae, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    # Initialise the temperature so the softmax logits start with unit spread
    # on THIS basis, measured on a sample of the training posts rather than
    # derived from d_sae. Learned from there.
    with torch.no_grad():
        probe_rows = np.concatenate(
            [np.arange(*bounds[i]) for i in range(min(64, len(bounds)))])
        raw = Z[probe_rows] @ q.detach().cpu().numpy()
        spread = float(np.std(raw)) or 1.0
    log_temp = torch.tensor(-np.log(spread), dtype=torch.float32,
                            device=dev, requires_grad=True)
    # The readout is L1-penalised and the query, until now, was not penalised
    # at all -- d_sae free parameters deciding where to look, fitted by the
    # same 500 steps. `q_l2` is decoupled weight decay on the query alone;
    # 0.0 reproduces every published row exactly.
    opt = torch.optim.Adam(
        [{"params": [q], "weight_decay": q_l2},
         {"params": [w, b, log_temp], "weight_decay": 0.0}], lr=ap["lp_lr"])
    lossf = torch.nn.BCEWithLogitsLoss()

    batch = int(ap.get("lp_batch", 128))
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for _ in range(int(ap["lp_steps"])):
        if len(order) < batch:
            order += rng.permutation(len(train_ids)).tolist()
        idx, order = order[:batch], order[batch:]
        opt.zero_grad()
        if dev is not None and dev.type == "cuda":
            # The mat-vec runs over the FULL training stack and the batch is
            # gathered after. Rows outside the batch contribute nothing to the
            # loss, so their gradient is zero and this is the same minibatch
            # step -- but it is one large sparse kernel instead of slicing a
            # 14.6M-non-zero CSR on the host every iteration.
            sel = torch.as_tensor(idx, dtype=torch.long, device=dev)
            logits = _pooled_logits_cuda(Z_dev, Zt_dev, idx_all[sel],
                                         mask_all[sel], q, w, b, log_temp)
            loss = lossf(logits, y[sel])
        else:
            rows = np.concatenate([np.arange(*bounds[i]) for i in idx])
            Zb = Z[rows]
            Zbt = Zb.T.tocsr()
            at, bb = 0, []
            for i in idx:
                n = bounds[i][1] - bounds[i][0]
                bb.append((at, at + n))
                at += n
            logits = _pooled_logits(Zb, Zbt, bb, q, w, b, log_temp)
            loss = lossf(logits, y[idx])
        loss.backward()
        opt.step()
        # The L1 prox, applied AFTER the Adam step and to the readout only.
        # See the module docstring: adding l1*|w| to the loss instead leaves
        # every coefficient fractionally non-zero and makes n_nonzero_coef a
        # meaningless number.
        with torch.no_grad():
            thr = ap["lp_lr"] * l1
            w.copy_(torch.sign(w) * torch.clamp(w.abs() - thr, min=0.0))

    with torch.no_grad():
        if dev is not None and dev.type == "cuda":
            train_scores = _pooled_logits_cuda(
                Z_dev, Zt_dev, idx_all, mask_all, q, w, b, log_temp).cpu().numpy()
        else:
            train_scores = _pooled_logits(Z, Zt, bounds, q, w, b, log_temp).numpy()
    w_np = w.detach().cpu().numpy().astype(np.float64)
    model = {
        "kind": "sae_attention",
        "q": q.detach().cpu().numpy().astype(np.float64),
        "w": w_np,
        "b": float(b.detach().cpu()),
        "d_sae": d_sae,
        "temperature": float(torch.exp(log_temp.detach().cpu())),
        "l1": l1,
        "q_l2": q_l2,
        "solver": SOLVER_NOTE,
        "n_posts": len(train_ids),
        # The sparsity claim, measured on the fitted coefficients exactly as
        # sae_probe measures it -- an L1 penalty is not a sparse model.
        # Exactly zero after the prox, so this is a count and not a
        # thresholding choice.
        "n_nonzero_coef": int((w_np != 0.0).sum()),
        "train_accuracy": float(((train_scores > 0).astype(int) == y_np).mean()),
        "threshold": best_f1_threshold(train_scores, y_np),
    }
    return model


# ------------------------------------------------------------------ scoring


def alpha(model: dict, rows) -> np.ndarray:
    s = np.asarray(rows @ model["q"]).ravel() * model["temperature"]
    e = np.exp(s - s.max())
    return e / e.sum()


def post_score(model: dict, rows) -> float:
    """The decision function for one post's [T, d_sae] latent rows."""
    return float(alpha(model, rows) @ np.asarray(rows @ model["w"]).ravel()
                 + model["b"])


def token_contributions(model: dict, rows) -> np.ndarray:
    """alpha_t * (w . z_t): the additive terms that sum to the decision."""
    return alpha(model, rows) * np.asarray(rows @ model["w"]).ravel()


def score_posts(model: dict, token_latents, post_ids) -> np.ndarray:
    return np.array([post_score(model, token_latents.rows[p]) for p in post_ids])


def verify_decomposition(model: dict, rows, tol: float = 1e-8) -> float:
    """Check sum_t alpha_t (w . z_t) + b equals the post score; raise past tol."""
    terms = float(token_contributions(model, rows).sum()) + model["b"]
    score = post_score(model, rows)
    gap = abs(terms - score) / max(abs(score), 1e-9)
    if gap > tol:
        raise AssertionError(
            f"attention decomposition off by {gap:.2e} (relative)")
    return gap
