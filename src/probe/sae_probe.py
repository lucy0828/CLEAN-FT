"""SAE probing: a sparse linear probe over a frozen, pretrained SAE dictionary.

Not run directly — `src.stage1.train` and `src.stage2.dla_auprc` drive it.

WHAT THIS METHOD IS
-------------------
Two other readouts of the same frozen residual stream are already in the
comparison: a dense L2 probe on the raw mean-pooled post vector, and an
attention probe over the same raw basis. This module adds a third point on
that axis — a basis the model itself was shown to use, learned without any
supervision from this corpus at all:

    1. project every post token onto the pinned Gemma Scope dictionary
       (`configs/default.yaml` -> sae.gemma-2-2b), giving a 16,384-d
       non-negative, sparse latent vector per token;
    2. mean-pool tokens to a post vector;
    3. fit an L1-regularised logistic regression on those post vectors.

The question the arm answers is whether the SAE's learned basis beats the raw
activations it was trained to reconstruct. Nothing here is tuned on an
outcome: the dictionary is pinned in config, `C` is a CLI argument reported on
every output row, and the pooling is declared up front.

THE SPARSITY CLAIM IS MEASURED, NOT ASSUMED
-------------------------------------------
"Sparse probe" is a claim about the fitted model, not about the penalty. So
`fit_sae_probe` counts the non-zero coefficients and returns the count, and the
runner writes it on every row. A probe with 14,000 non-zero coefficients out of
16,384 is a dense probe wearing an L1 penalty, and the table should say so.

THE ADDITIVE DECOMPOSITION
--------------------------
Under mean pooling the post logit is exactly the mean of per-token scores:

    score(post) = w . mean_t f(X_t) + b
                = mean_t (w . f(X_t)) + b
                = mean_t token_score(t) + b

so `token_scores_sae_probe` returns terms that reconstruct the post score with
no residual and no approximation — the same kind of identity the label probe
and the attention probe rely on. `verify_additive_identity` checks it rather
than trusting it.

NO LABEL TOUCHES THE FEATURES
-----------------------------
The dictionary is pretrained and frozen; encoding reads token vectors and
nothing else — no label, no fold, no gold span, not even the corpus's own
statistics (there is no fitted scaler by default, and `--standardize` fits its
scale on training folds alone). Feature construction is therefore fold-blind
and is done once for the whole corpus, exactly like the frozen residual
features in `src.probe`.

WHICH TOKENS ARE ENCODED
------------------------
Every token in the layer cache, and only those. `src.utils.data_prep`
keeps a token only when it overlaps the post's own character range
(`a < p1 and b > p0 and b > a`, with offsets rebased onto the post text), so
chat-template and padding tokens never reach the parquet and cannot reach the
SAE. Verified by reading that filter, not assumed.

A NOTE ON THE LAYER INDEX
-------------------------
The pinned Gemma SAE reports `hook_name = blocks.17.hook_resid_post` (repinned
from block 12 on 2026-08-26; see the `sae:` block in configs/default.yaml). The
cache is written from HuggingFace `output_hidden_states`, where
`hidden_states[k]` is the *input* to block k for k < depth, i.e.
`blocks.k.hook_resid_pre`. Those two sites differ by one block, so the SAE's
own layer number and the cached layer it belongs on are NOT the same integer:

    SAE at blocks.17.hook_resid_post
      == input to block 18
      == hidden_states[18]
      == cached layer 18            (17 + SAE_CACHE_OFFSET)

`SAE_PROBE_LAYER["gemma-2-2b"] == 18` therefore reads the SAE at layer 17, and
reading cached layer 17 instead would evaluate the dictionary one block early,
against `blocks.16.hook_resid_post`. Nothing would raise; the latents would
just be wrong.

Which cached layer the SAE belongs on stays an empirical question, not a naming
one. `reconstruction_diagnostics` measures it: a dictionary reconstructs well
at the site it was trained on and poorly one block away, so run it across two
adjacent cached layers before believing any layer label.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import torch
from hashlib import sha256
from sklearn.metrics import average_precision_score
from typing import Any

from src.utils.data_prep import CACHE, build_prompt, words

# Re-exported so this module is the single import surface for SAE probing:
# src/probe/__init__.py's _sae_profile() hands callers this module and they
# dereference TokenAlignment/aggregate_word_scores off it, and load_pinned_sae
# below calls load_pretrained_sae. All three live in src.utils.sae_features.
from src.utils.sae_features import (  # noqa: F401
    TokenAlignment,
    aggregate_word_scores,
    load_pretrained_sae,
)

# Token rows pushed through sae.encode() at once. The per-token latent block is
# [batch, 16384] float32 = 128 MB at 2048, which is the point of batching: the
# whole corpus as one dense [505k, 16384] block would be 31 GB.
DEFAULT_TOKEN_BATCH = 2048

# Token rows the reconstruction diagnostic samples. It exists to catch a
# layer/site mismatch, which shows up in the first few thousand tokens or not
# at all, so it does not need the corpus.
DIAGNOSTIC_TOKENS = 8192

# Mean only. The study reads a probe's word scores as its decision, which
# needs the exact mean-pooling identity below; the `pool` field and the cache
# filename suffix stay so the encoded latent caches on disk keep their names.
POOLINGS = ("mean",)


def _l1_kwargs() -> dict:
    """How to ask this sklearn for a pure L1 penalty.

    scikit-learn 1.8 deprecated `penalty=` in favour of `l1_ratio=`, and 1.10
    removes it. Passing `penalty="l1"` on a new sklearn still gives an L1 fit
    (verified: identical non-zero counts to `l1_ratio=1.0`) but pairs it with
    the default `l1_ratio=0.0` and warns about the inconsistency — exactly the
    shape of thing that later flips silently to L2 and turns "sparse probe"
    into a false claim. So the spelling is chosen from the installed version
    rather than hard-coded, and `n_nonzero_coef` measures the outcome either way.
    """
    import inspect

    default = inspect.signature(LogisticRegression.__init__).parameters[
        "penalty"].default
    return {"l1_ratio": 1.0} if default == "deprecated" else {"penalty": "l1"}


L1_KWARGS = _l1_kwargs()


class SAEProbeError(RuntimeError):
    """A stop that names what is wrong with the inputs, not a deep traceback."""


# --------------------------------------------------------------------- paths


def activation_cache_path(model: str, corpus: str, layer: int) -> Path:
    """The token-activation parquet this module reads (written by Stage 2)."""
    return CACHE / model / f"{corpus}_layer{layer:02d}.parquet"


def latent_cache_path(model: str, corpus: str, layer: int, pool: str) -> Path:
    """Post-level SAE latents: the expensive pass runs once and is reloaded."""
    return CACHE / model / f"{corpus}_layer{layer:02d}_sae_{pool}.npz"


def token_latent_cache_path(model: str, corpus: str, layer: int, tag: str) -> Path:
    """Per-token SAE latents for a named subset (the plausibility posts).

    Per-token latents for the whole corpus would be ~500k x 153 non-zeros;
    only the posts that are actually scored token-by-token are kept.
    """
    return CACHE / model / f"{corpus}_layer{layer:02d}_sae_tokens_{tag}.npz"


# ------------------------------------------------------------------- records


@dataclass
class PostLatents:
    """Post-level SAE features for a corpus, plus what was measured making them.

    `X` is [n_posts, d_sae] CSR: row i is the pooled latent vector of
    `post_ids[i]`. `diagnostics` carries the numbers that decide whether the
    features are trustworthy at all (reconstruction error, latent density),
    measured during the encode pass and cached with the matrix.
    """

    post_ids: list[str]
    X: sparse.csr_matrix
    pool: str
    d_sae: int
    n_tokens: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def index_of(self) -> dict[str, int]:
        return {pid: i for i, pid in enumerate(self.post_ids)}

    @property
    def density(self) -> float:
        """Fraction of the post x latent matrix that is non-zero."""
        n = self.X.shape[0] * self.X.shape[1]
        return float(self.X.nnz / n) if n else float("nan")


@dataclass
class TokenLatents:
    """Per-token SAE latents for a subset of posts, keyed by post id.

    `rows[pid]` is [T_pid, d_sae] CSR in cached-token order, and
    `offsets[pid]` is the matching [T_pid, 2] character span array, so a token
    score can be laid back onto the post text without re-reading the parquet.
    """

    rows: dict[str, sparse.csr_matrix]
    offsets: dict[str, np.ndarray]
    d_sae: int

    def __contains__(self, pid: str) -> bool:
        return pid in self.rows


@dataclass
class SAEProbe:
    """A fitted L1 logistic probe, with the sparsity claim already counted."""

    clf: LogisticRegression
    C: float
    seed: int
    pool: str
    d_sae: int
    scale: np.ndarray | None
    n_nonzero_coef: int
    n_train: int
    solver: str

    def effective_affine(self) -> tuple[np.ndarray, float]:
        """(w, b) with logit(x) == x @ w + b for a raw latent vector x.

        Any train-fold standardisation is folded into `w` here, so every
        downstream identity is stated on raw latents and cannot drift out of
        step with whatever preprocessing the fit used.
        """
        w = self.clf.coef_[0].astype(np.float64)
        if self.scale is not None:
            w = w / self.scale
        return w, float(self.clf.intercept_[0])


# ------------------------------------------------------------------ encoding


def resolve_device(device: str | None = None) -> str:
    """Requested device, else GPU when one is actually usable, else CPU."""
    import torch

    if device:
        return device
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # a broken driver is a CPU run, not a crash
        return "cpu"


def _read_token_batches(
    path: Path, batch_rows: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream (post_id, [n,2] offsets, [n,H] float32 vectors) from the parquet.

    Streaming is the whole point: the corpus is ~505k token rows of 2304
    float16, and the SAE turns each into 16,384 floats. Nothing bigger than one
    batch is ever dense.
    """
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(
        batch_size=batch_rows, columns=["post_id", "start", "end", "vec"]
    ):
        vec = batch.column("vec")
        hidden = vec.type.list_size
        flat = vec.flatten().to_numpy(zero_copy_only=False)
        yield (
            batch.column("post_id").to_numpy(zero_copy_only=False),
            np.stack(
                [
                    batch.column("start").to_numpy(zero_copy_only=False),
                    batch.column("end").to_numpy(zero_copy_only=False),
                ],
                axis=1,
            ).astype(np.int64),
            flat.reshape(len(vec), hidden).astype(np.float32),
        )


def _sample_row_groups(
    path: Path, n_tokens: int, batch_rows: int
) -> Iterator[np.ndarray]:
    """Token vectors from row groups spread across the whole file.

    The activation cache is written in length-sorted post order, so reading
    from the start samples only the shortest posts. Striding the row groups
    keeps a diagnostic honest about the corpus it claims to describe.
    """
    handle = pq.ParquetFile(path)
    n_groups = handle.metadata.num_row_groups
    per_group = max(1, handle.metadata.num_rows // max(n_groups, 1))
    wanted = max(1, min(n_groups, -(-n_tokens // per_group)))
    step = max(1, n_groups // wanted)
    taken = 0
    for group in range(0, n_groups, step):
        if taken >= n_tokens:
            return
        table = handle.read_row_group(group, columns=["vec"])
        vec = table.column("vec").combine_chunks()
        block = (
            vec.flatten().to_numpy(zero_copy_only=False)
            .reshape(len(vec), vec.type.list_size).astype(np.float32)
        )
        for start in range(0, len(block), batch_rows):
            if taken >= n_tokens:
                return
            chunk = block[start: start + batch_rows]
            taken += len(chunk)
            yield chunk


def _encode_block(sae, vectors: np.ndarray, device: str) -> np.ndarray:
    """One batch of token vectors -> [n, d_sae] dense float32 latents."""
    import torch

    with torch.no_grad():
        x = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
        return sae.encode(x).detach().to("cpu").numpy().astype(np.float32)


def reconstruction_diagnostics(
    sae,
    model: str,
    corpus: str,
    layer: int,
    *,
    device: str | None = None,
    n_tokens: int = DIAGNOSTIC_TOKENS,
    batch_rows: int = DEFAULT_TOKEN_BATCH,
) -> dict:
    """Does this SAE actually belong on this cached layer? Measure, do not assume.

    Returns the relative L2 reconstruction error ||x - dec(enc(x))|| / ||x||
    and the mean number of active latents per token, over a sample of cached
    tokens. A dictionary evaluated at the residual-stream site it was trained
    on reconstructs well; one block away it does not, so running this across
    two adjacent cached layers settles the HuggingFace-vs-TransformerLens
    indexing question empirically (see the module docstring).
    """
    import torch

    path = activation_cache_path(model, corpus, layer)
    if not path.exists():
        raise SAEProbeError(f"no cached activations at {path}")
    device = resolve_device(device)
    errors: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    active: list[np.ndarray] = []
    seen = 0
    # Evenly spaced row groups, not the first N rows: the cache is written in
    # length-sorted post order, so the head of the file is the shortest posts
    # in the corpus and would make this a diagnostic of short posts only.
    for vectors in _sample_row_groups(path, n_tokens, batch_rows):
        if seen >= n_tokens:
            break
        vectors = vectors[: max(0, n_tokens - seen)]
        with torch.no_grad():
            x = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
            latents = sae.encode(x)
            recon = sae.decode(latents)
            errors.append((recon - x).pow(2).sum(-1).sqrt().cpu().numpy())
            scales.append(x.pow(2).sum(-1).sqrt().cpu().numpy())
            active.append((latents > 0).sum(-1).float().cpu().numpy())
        seen += len(vectors)
    if not errors:
        raise SAEProbeError(f"{path} holds no token rows to diagnose")
    err = np.concatenate(errors)
    scale = np.concatenate(scales)
    rel = err / np.maximum(scale, 1e-12)
    return {
        "layer": int(layer),
        "n_tokens_sampled": int(seen),
        "reconstruction_rel_l2_mean": float(rel.mean()),
        "reconstruction_rel_l2_median": float(np.median(rel)),
        "mean_active_latents_per_token": float(np.concatenate(active).mean()),
    }


def encode_posts_to_latents(
    sae,
    model: str,
    corpus: str,
    layer: int,
    *,
    post_ids: Sequence[str],
    pool: str = "mean",
    device: str | None = None,
    batch_rows: int = DEFAULT_TOKEN_BATCH,
    token_post_ids: Iterable[str] | None = None,
    token_tag: str | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> tuple[PostLatents, TokenLatents | None]:
    """Project cached token activations onto the SAE and pool them to posts.

    One streaming pass over the parquet does both jobs: it accumulates the
    post-level pooled latents for every post in `post_ids`, and — when
    `token_post_ids` is given — keeps the full per-token latent rows for that
    (much smaller) subset, which is what the token-level plausibility readout
    needs. Both results are cached, so the GPU pass runs once.

    Memory: the accumulator is one dense [n_posts, d_sae] float32 array
    (~230 MB at 3,522 x 16,384); the per-token intermediate is one batch wide
    and is never materialised for the corpus.

    `pool` is "mean" — mean latent activation over the post's cached content
    tokens, the pooling the additive token decomposition is exact under (see
    `token_scores_sae_probe`). It stays a parameter because it names the latent
    cache file on disk.
    """
    if pool not in POOLINGS:
        raise SAEProbeError(f"unknown pooling {pool!r}; expected one of {POOLINGS}")
    post_ids = list(post_ids)
    want_tokens = token_post_ids is not None
    if want_tokens and not token_tag:
        raise SAEProbeError("token_post_ids needs a token_tag to cache under")

    post_path = latent_cache_path(model, corpus, layer, pool)
    tok_path = (
        token_latent_cache_path(model, corpus, layer, token_tag) if want_tokens else None
    )
    if not refresh and post_path.exists() and (not want_tokens or tok_path.exists()):
        cached = load_post_latents(post_path)
        tokens = load_token_latents(tok_path) if want_tokens else None
        if cached.post_ids == post_ids and (
            not want_tokens or set(token_post_ids) <= set(tokens.rows)
        ):
            if verbose:
                print(f"  reusing cached SAE latents: {post_path.name}"
                      + (f" + {tok_path.name}" if want_tokens else ""))
            return cached, tokens
        if verbose:
            print("  cached SAE latents do not cover the requested posts; "
                  "re-encoding")

    act_path = activation_cache_path(model, corpus, layer)
    if not act_path.exists():
        raise SAEProbeError(f"no cached activations at {act_path}")
    if sae is None:
        raise SAEProbeError(
            "the cached latents do not cover this request, so the encode "
            "pass has to run — but no SAE was passed. Re-run with the "
            "dictionary loaded (--allow-model-process)."
        )

    device = resolve_device(device)
    d_sae = int(sae.W_dec.shape[0])
    d_in = int(sae.W_dec.shape[1])
    index_of = {pid: i for i, pid in enumerate(post_ids)}
    wanted_tokens = set(token_post_ids) if want_tokens else set()

    # The one dense object in this function, and the reason it is affordable.
    acc = np.zeros((len(post_ids), d_sae), dtype=np.float32)
    counts = np.zeros(len(post_ids), dtype=np.int64)
    tok_rows: dict[str, list[sparse.csr_matrix]] = {p: [] for p in wanted_tokens}
    tok_offsets: dict[str, list[np.ndarray]] = {p: [] for p in wanted_tokens}
    active_total, seen, skipped = 0.0, 0, 0
    t0 = time.time()

    for ids, offsets, vectors in _read_token_batches(act_path, batch_rows):
        if vectors.shape[1] != d_in:
            raise SAEProbeError(
                f"cached vectors are dim {vectors.shape[1]} but the SAE takes "
                f"{d_in}; the dictionary and the layer cache disagree."
            )
        keep = np.array([pid in index_of for pid in ids])
        if not keep.any():
            skipped += len(ids)
            continue
        if not keep.all():
            skipped += int((~keep).sum())
            ids, offsets, vectors = ids[keep], offsets[keep], vectors[keep]

        latents = _encode_block(sae, vectors, device)
        active_total += float((latents > 0).sum())
        seen += len(latents)

        rows = np.array([index_of[pid] for pid in ids])
        uniq, inverse = np.unique(rows, return_inverse=True)
        for slot, row in enumerate(uniq):
            block = latents[inverse == slot]
            acc[row] += block.sum(axis=0)
            counts[row] += len(block)

        if wanted_tokens:
            for slot, row in enumerate(uniq):
                pid = post_ids[row]
                if pid not in wanted_tokens:
                    continue
                mask = inverse == slot
                tok_rows[pid].append(sparse.csr_matrix(latents[mask]))
                tok_offsets[pid].append(offsets[mask])

        if verbose and seen % (batch_rows * 25) < batch_rows:
            rate = seen / max(time.time() - t0, 1e-9)
            print(f"  encoded {seen} tokens ({rate:,.0f}/s)", flush=True)

    if counts.sum() == 0:
        raise SAEProbeError(
            f"{act_path} holds no rows for any of the {len(post_ids)} requested "
            "posts; the cache and posts.csv disagree."
        )
    empty = [post_ids[i] for i in np.flatnonzero(counts == 0)]
    if empty:
        raise SAEProbeError(
            f"{len(empty)} posts have no cached tokens at layer {layer} "
            f"(first: {empty[0]}); the cache is incomplete for this corpus."
        )
    acc /= counts[:, None].astype(np.float32)

    latents_csr = sparse.csr_matrix(acc)
    del acc
    diagnostics = {
        "n_tokens_encoded": int(seen),
        "n_token_rows_skipped": int(skipped),
        "mean_active_latents_per_token": float(active_total / max(seen, 1)),
        "device": device,
        "d_sae": d_sae,
        "d_in": d_in,
        "encode_seconds": float(time.time() - t0),
    }
    out = PostLatents(
        post_ids=post_ids,
        X=latents_csr,
        pool=pool,
        d_sae=d_sae,
        n_tokens=counts,
        diagnostics=diagnostics,
    )
    save_post_latents(post_path, out)

    tokens = None
    if wanted_tokens:
        tokens = TokenLatents(
            rows={
                pid: sparse.vstack(parts, format="csr")
                for pid, parts in tok_rows.items()
                if parts
            },
            offsets={
                pid: np.concatenate(parts, axis=0)
                for pid, parts in tok_offsets.items()
                if parts
            },
            d_sae=d_sae,
        )
        save_token_latents(tok_path, tokens)
    if verbose:
        print(f"  encoded {seen} tokens in {diagnostics['encode_seconds']:.1f}s "
              f"on {device}; post-latent density {out.density:.4f}, "
              f"{diagnostics['mean_active_latents_per_token']:.1f} active "
              "latents per token")
    return out, tokens


# --------------------------------------------------------------------- cache


def save_post_latents(path: Path, latents: PostLatents) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X = latents.X.tocsr()
    np.savez(
        path,
        data=X.data.astype(np.float32),
        indices=X.indices,
        indptr=X.indptr,
        shape=np.asarray(X.shape),
        post_ids=np.asarray(latents.post_ids, dtype=object),
        n_tokens=latents.n_tokens,
        pool=np.asarray(latents.pool),
        diagnostics=np.asarray(json.dumps(latents.diagnostics)),
    )


def load_post_latents(path: Path) -> PostLatents:
    with np.load(path, allow_pickle=True) as z:
        X = sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        return PostLatents(
            post_ids=[str(p) for p in z["post_ids"]],
            X=X,
            pool=str(z["pool"]),
            d_sae=int(X.shape[1]),
            n_tokens=z["n_tokens"],
            diagnostics=json.loads(str(z["diagnostics"])),
        )


def save_token_latents(path: Path, tokens: TokenLatents) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(tokens.rows)
    stacked = sparse.vstack([tokens.rows[p] for p in ids], format="csr")
    bounds = np.cumsum([0] + [tokens.rows[p].shape[0] for p in ids])
    np.savez(
        path,
        data=stacked.data.astype(np.float32),
        indices=stacked.indices,
        indptr=stacked.indptr,
        shape=np.asarray(stacked.shape),
        post_ids=np.asarray(ids, dtype=object),
        bounds=bounds,
        offsets=np.concatenate([tokens.offsets[p] for p in ids], axis=0),
    )


def load_token_latents(path: Path) -> TokenLatents:
    with np.load(path, allow_pickle=True) as z:
        stacked = sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        ids = [str(p) for p in z["post_ids"]]
        bounds = z["bounds"]
        offsets = z["offsets"]
        return TokenLatents(
            rows={
                pid: stacked[bounds[i]: bounds[i + 1]] for i, pid in enumerate(ids)
            },
            offsets={
                pid: offsets[bounds[i]: bounds[i + 1]] for i, pid in enumerate(ids)
            },
            d_sae=int(stacked.shape[1]),
        )


# ---------------------------------------------------------------- the probe


def fit_sae_probe(
    X_train,
    y_train,
    seed: int,
    C: float,
    *,
    pool: str = "mean",
    solver: str = "liblinear",
    max_iter: int = 2000,
    standardize: bool = False,
) -> SAEProbe:
    """L1 logistic regression on pooled SAE latents.

    L1 rather than L2 because the claim being tested is that a *small* set of
    dictionary features carries the signal; an L2 probe over 16,384 latents
    would be a dense readout of a sparse basis and would say nothing about
    which features matter. `n_nonzero_coef` on the returned model is what turns
    that from a claim into a measurement.

    No centering. Centering a sparse latent matrix densifies it, and — more to
    the point — the interpretable object here is "coefficient times the
    activation this feature actually had", which a subtracted mean would turn
    into "times its deviation from the corpus mean". `standardize=True` divides
    by the train-fold standard deviation only (never centres), and the scale is
    folded back into `effective_affine`, so the token identity is unaffected.
    """
    if solver not in {"liblinear", "saga"}:
        raise SAEProbeError(f"unknown solver {solver!r}; expected liblinear or saga")
    X = sparse.csr_matrix(X_train)
    y = np.asarray(y_train).astype(int)
    if len(np.unique(y)) < 2:
        raise SAEProbeError("training labels are single-class; a probe cannot fit")

    scale = None
    if standardize:
        # E[x^2] - E[x]^2 computed on the sparse matrix, no densification.
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        scale = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
        scale[scale == 0] = 1.0
        X = (X @ sparse.diags(1.0 / scale)).tocsr()

    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
        **L1_KWARGS,
    )
    clf.fit(X, y)
    return SAEProbe(
        clf=clf,
        C=float(C),
        seed=int(seed),
        pool=pool,
        d_sae=int(X.shape[1]),
        scale=scale,
        n_nonzero_coef=int(np.count_nonzero(clf.coef_[0])),
        n_train=int(X.shape[0]),
        solver=solver,
    )


def _apply_scale(model: SAEProbe, X) -> sparse.csr_matrix:
    X = sparse.csr_matrix(X)
    if model.scale is None:
        return X
    return (X @ sparse.diags(1.0 / model.scale)).tocsr()


def score_sae_probe(model: SAEProbe, X) -> np.ndarray:
    """P(present) for each row of pooled latents."""
    return model.clf.predict_proba(_apply_scale(model, X))[:, 1]


def logit_sae_probe(model: SAEProbe, X) -> np.ndarray:
    """The decision function — the quantity the token scores decompose."""
    return model.clf.decision_function(_apply_scale(model, X))


def token_scores_sae_probe(model: SAEProbe, token_latents) -> np.ndarray:
    """Per-token contribution: sum over latents of coefficient x activation.

    For token t with latent vector f(X_t), the score is

        token_score(t) = sum_j w_j * f(X_t)_j = w . f(X_t)

    Under MEAN pooling this is the exact additive decomposition of the post
    score, with no residual term:

        score(post) = w . mean_t f(X_t) + b = mean_t (w . f(X_t)) + b

    so the post logit is the plain mean of the returned token scores plus the
    intercept. `verify_additive_identity` checks that on real data rather than
    taking the algebra on trust.

    Returns one score per row of `token_latents`, in cached-token order.
    """
    w, _ = model.effective_affine()
    X = sparse.csr_matrix(token_latents)
    if X.shape[1] != len(w):
        raise SAEProbeError(
            f"token latents are width {X.shape[1]} but the probe has "
            f"{len(w)} coefficients"
        )
    if model.pool != "mean":
        raise SAEProbeError(f"unknown pooling {model.pool!r}")
    return np.asarray(X @ w, dtype=np.float64).ravel()


def verify_additive_identity(
    model: SAEProbe, token_latents, *, atol: float = 1e-4
) -> float:
    """|mean(token scores) + b - logit(pooled post)|; raises if broken.

    Mean pooling makes the post logit the MEAN of the token scores, exactly.
    """
    if model.pool not in POOLINGS:
        raise SAEProbeError(f"unknown pooling {model.pool!r}")
    X = sparse.csr_matrix(token_latents)
    _, b = model.effective_affine()
    pooled = sparse.csr_matrix(np.asarray(X.mean(axis=0)))
    from_tokens = float(token_scores_sae_probe(model, X).mean() + b)
    direct = float(logit_sae_probe(model, pooled)[0])
    err = abs(direct - from_tokens)
    if err > atol:
        raise SAEProbeError(
            f"additive identity violated: post logit {direct:.6f} but the "
            f"token decomposition gives {from_tokens:.6f} (|diff| {err:.2e})"
        )
    return err


def word_scores_sae_probe(
    token_scores: np.ndarray,
    offsets: np.ndarray,
    text: str,
    *,
    reduction: str = "max",
) -> np.ndarray:
    """Lay token scores onto the study's word units via character offsets.

    Words come from `src.utils.data_prep.words`, the one word tokenizer every method
    in the study is scored over. A word takes the maximum over the tokens whose
    character span overlaps it (`audit.word_score_aggregation` in config, and
    the same rule `aggregate_word_scores` below applies); words
    no token touches keep NaN so they rank last rather than pretending to a
    score of zero.

    Note that max over tokens breaks the additive identity by construction —
    the exact decomposition lives at the token level, and this is the
    presentation layer on top of it.
    """
    if reduction not in {"max", "mean", "sum"}:
        raise SAEProbeError(f"unknown word score reduction {reduction!r}")
    token_scores = np.asarray(token_scores, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(token_scores) != len(offsets):
        raise SAEProbeError(
            f"{len(token_scores)} token scores against {len(offsets)} offsets"
        )
    post_words = words(text)
    out = np.full(len(post_words), np.nan, dtype=np.float64)
    if not len(token_scores):
        return out
    starts, ends = offsets[:, 0], offsets[:, 1]
    for i, word in enumerate(post_words):
        mask = (starts < word["end"]) & (ends > word["start"])
        if mask.any():
            values = token_scores[mask]
            out[i] = float(
                np.max(values) if reduction == "max"
                else np.mean(values) if reduction == "mean"
                else np.sum(values)
            )
    return out


def load_pinned_sae(cfg: dict, model: str, *, device: str | None,
                    allow: bool, command: str):
    """Load exactly the pinned dictionary, offline, behind the study gate.

    The artifact is already downloaded; forcing HF_HUB_OFFLINE=1 means the
    gate guards a local read rather than a silent network fetch. Returns
    (sae, sae_config, device).
    """
    import os

    from src.utils.common import require_model_process_approval

    sae_config = cfg.get("sae", {}).get(model)
    if not sae_config or not sae_config.get("release"):
        raise SAEProbeError(
            f"configs/default.yaml pins no SAE for {model!r}; SAE probing has "
            "no dictionary to project onto.")
    require_model_process_approval(cfg, cli_override=allow, command=command)

    device = resolve_device(device)
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        sae = load_pretrained_sae(sae_config, device=device)
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
    print(f"SAE {sae_config['release']} / {sae_config['sae_id']} on {device}: "
          f"d_sae={sae.W_dec.shape[0]}, d_in={sae.W_dec.shape[1]}")
    return sae, sae_config, device
