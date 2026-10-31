"""The two SAE probes' readouts as functions of raw residual vectors.

Not run directly -- `src.stage2.word_deletion --compute` drives it.

WHY THIS EXISTS
---------------
Word deletion re-encodes shortened TEXT, so every probe it audits has to expose
its readout as a function of a fresh [T, H] token matrix (`Fitted.score_vectors`).
The dense probes in `probing.FITTERS` do. The two SAE probes do not, and cannot:
they are not functions of the residual stream at all. They fit on SAE LATENTS at
`SAE_PROBE_LAYER`, one block below the `DENSE_PROBE_LAYER` the dense audit
re-encodes at, and their coefficients index a dictionary rather than a hidden
dimension. That is why they carried no row in this metric until 2026-08-28.

This module supplies the missing function. The composition is

    text -> FrozenBackbone(SAE_PROBE_LAYER) -> [T, H]
         -> pinned SAE encoder             -> [T, d_sae]
         -> the probe's own readout        -> one logit

so the deleted post goes through the SAME dictionary the probe was fitted on,
at the layer it was fitted at. Nothing here ranks words by the dictionary: the
ranking is the probe's own token decomposition, exactly as it is for every other
method in `src.stage2.word_deletion`. The clinical-SAE-RANKED metric that did
rank by a manifest was removed on 2026-08-28 and is unrelated to this path.

WHY THE LATENTS ARE RECOMPUTED AND NOT LOOKED UP
------------------------------------------------
A deleted post is not in any latent cache -- that is the whole point of the
intervention -- so `encode_posts_to_latents` cannot serve it. The SAE encoder is
run directly on the fresh vectors instead, which is what the cache builder does
internally anyway.

THE TWO READOUTS DIFFER IN WHAT A TOKEN CONTRIBUTES
---------------------------------------------------
`sae_probe` mean-pools, so a token's contribution is the intrinsic `w . z_t` and
the post logit is their plain mean plus the intercept. `sae_attention_probe`
pools with a learned query, so a token contributes `alpha_t (w . z_t)` and what
it is worth depends on what else the post says. Both are the additive
decompositions their own modules guarantee; neither is re-derived here.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class SAEFrozenReadout:
    """One fitted SAE probe, scored on arbitrary residual vectors.

    Mirrors the two callables `src.probe.Fitted` exposes for the dense probes:
    `score_vectors` for the post logit and `token_vectors_scores` for the token
    decomposition, both taking a [T, H] residual matrix at `SAE_PROBE_LAYER`.
    """

    def __init__(self, kind: str, probe, sae, device: str):
        if kind not in ("sae_probe", "sae_attention_probe"):
            raise ValueError(f"unknown SAE readout {kind!r}")
        self.kind = kind
        self.probe = probe
        self.sae = sae
        self.device = device

    # ------------------------------------------------------------- encoding
    def latents(self, vectors) -> np.ndarray:
        """[T, H] residual vectors -> [T, d_sae] dense latents."""
        from src.probe.sae_probe import _encode_block

        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected a [T, H] matrix, got {x.shape}")
        if not len(x):
            return np.zeros((0, int(self.sae.W_dec.shape[0])), dtype=np.float32)
        return _encode_block(self.sae, x, self.device)

    # -------------------------------------------------------------- readout
    def score_vectors(self, vectors) -> float:
        """The probe's logit for a post made of these token vectors.

        The EMPTY endpoint the caller needs (`score_vectors` of a single zero
        row) falls out correctly without a special case: a zero residual vector
        encodes to whatever the dictionary maps it to, is pooled alone, and the
        readout returns its value on that one row -- the same convention the
        dense probes' `score_vectors` follows.
        """
        from src.probe import sae_attention_probe
        from src.probe.sae_probe import logit_sae_probe

        z = self.latents(vectors)
        if self.kind == "sae_probe":
            pooled = z.mean(axis=0, keepdims=True) if len(z) else \
                np.zeros((1, z.shape[1]), dtype=np.float32)
            return float(logit_sae_probe(self.probe, pooled)[0])
        return float(sae_attention_probe.post_score(
            self.probe, sparse.csr_matrix(z)))

    def token_vectors_scores(self, vectors) -> np.ndarray:
        """One additive contribution per token vector."""
        from src.probe import sae_attention_probe
        from src.probe.sae_probe import token_scores_sae_probe

        z = self.latents(vectors)
        if not len(z):
            return np.zeros(0, dtype=np.float64)
        rows = sparse.csr_matrix(z)
        if self.kind == "sae_probe":
            return token_scores_sae_probe(self.probe, rows)
        return np.asarray(sae_attention_probe.token_contributions(
            self.probe, rows), dtype=np.float64)


def fit_readouts(kind: str, model: str, folds, seed: int, cfg: dict,
                 allow: bool):
    """{fold: SAEFrozenReadout} plus the cached activations to audit against.

    Fitting reads the corpus latent caches the study already builds -- the
    pooled cache for `sae_probe`, the whole-corpus token cache for
    `sae_attention_probe`, which fits its query over tokens -- so a cell here is
    the same fit `src.utils.dla` scores for plausibility, not a second one.
    """
    from src.probe import sae_attention_probe, sae_probe
    from src.utils.common import SAE_PROBE_LAYER
    from src.utils.data_prep import load_corpus

    layer = SAE_PROBE_LAYER[model]
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    y = (posts["label"] == "present").to_numpy().astype(int)
    fold_col = posts["fold"].to_numpy()

    # `sae_attention_probe` needs per-token latents for every TRAINING post,
    # not just the audited subset, so both kinds read the whole-corpus token
    # cache and the pooled cache is taken from the same call.
    have = (sae_probe.latent_cache_path(model, "lonxplain", layer,
                                        "mean").exists()
            and sae_probe.token_latent_cache_path(
                model, "lonxplain", layer, "all").exists())
    sae_for_cache = None
    if not have:
        sae_for_cache, _, _ = sae_probe.load_pinned_sae(
            cfg, model, device=None, allow=allow,
            command="src.stage2.word_deletion --compute")
    latents, token_latents = sae_probe.encode_posts_to_latents(
        sae_for_cache, model, "lonxplain", layer, post_ids=all_ids,
        pool="mean", token_post_ids=all_ids, token_tag="all")

    # The encoder itself is needed live regardless of the caches: deleted text
    # is not in any of them.
    sae, _, device = sae_probe.load_pinned_sae(
        cfg, model, device=None, allow=allow,
        command="src.stage2.word_deletion --compute")

    out = {}
    for fold in folds:
        train = np.flatnonzero(fold_col != fold)
        if kind == "sae_probe":
            fitted = sae_probe.fit_sae_probe(latents.X[train], y[train],
                                             seed=seed, C=1.0, pool="mean")
        else:
            fitted = sae_attention_probe.fit(
                token_latents, posts, [all_ids[i] for i in train], seed, cfg)
        out[fold] = SAEFrozenReadout(kind, fitted, sae, device)
        print(f"  {kind} fold {fold}: fitted", flush=True)
    return out, posts, layer
