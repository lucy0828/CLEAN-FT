"""Re-encoding frozen-probe word deletions -> the probe half of word_deletion.

Not run directly -- `src.stage2.word_deletion --compute` drives it.

Word deletion for a PROBE cannot reuse the activation cache: deleting words
changes the text, so the post has to go back through the backbone to get new
token vectors. That is the whole reason this module exists and the reason the
probe half of word deletion needs a GPU at all.

`FrozenBackbone` is restored verbatim from the engine removed on 2026-08-26
(~/lonxplain_removed_saefaith_20260826). It reproduces
`src.utils.data_prep.encode_corpus` decision for decision -- same chat template,
same `add_special_tokens=False`, same truncation, same "keep a token only if it
overlaps the post's character range" filter, same float16 round-trip -- because
a deletion logit is only comparable to the clean logit if both activations came
off the same pipeline. `cache_parity` measures whether that held, and it is
written on every row rather than assumed.

What is NOT restored is the SAE knockout the old engine wrapped this in. A probe
is scored here by its own `score_vectors`, so no dictionary is involved.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.utils.data_prep import MODEL_REGISTRY, build_prompt, words
from src.utils.faithfulness import (
    _add_aopc,
    delete_words,
    deterministic_rng,
    k_word_count,
    normalized_logit_decay,
)


def _fail(message: str) -> SystemExit:
    return SystemExit(message)


class FrozenBackbone:
    """The corpus cache's own encoder, re-run on word-deletion variants.

    Reproduces `src.utils.data_prep.encode_corpus` decision for
    decision -- the same chat template via `build_prompt`, the same
    `add_special_tokens=False`, the same truncation length read off the cache
    meta, the same "keep a token only if it overlaps the post's character
    range" filter, and the same float16 round-trip -- because a deletion logit
    is only comparable to the clean logit if the two activations came off the
    same pipeline. `cache_parity` measures whether that held.
    """

    def __init__(self, name: str, layer: int, max_len: int, model, tokenizer):
        self.name = name
        self.layer = layer
        self.max_len = max_len
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, name: str, layer: int, max_len: int) -> "FrozenBackbone":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if name not in MODEL_REGISTRY:
            raise _fail(f"no model id registered for {name!r}")
        model_id = MODEL_REGISTRY[name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        if layer > int(model.config.num_hidden_layers):
            raise _fail(
                f"cached layer {layer} is past {name}'s depth "
                f"({model.config.num_hidden_layers} blocks)"
            )
        return cls(name, layer, max_len, model, tokenizer)

    def _prepare(self, text: str) -> tuple[list[int], list[int]]:
        full, p0, p1 = build_prompt(self.tokenizer, text)
        enc = self.tokenizer(
            full,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )
        keep = [
            ti
            for ti, (a, b) in enumerate(enc["offset_mapping"])
            if a < p1 and b > p0 and b > a
        ]
        return enc["input_ids"], keep

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[np.ndarray]:
        """[T_i, H] residual states at the cached layer, one array per text.

        No texts is a legitimate state, not an error: a post on which no
        clinical feature fires produces an empty deletion plan. The check comes
        before the model is touched, so the empty case needs no GPU at all --
        this is the guard `_score_texts` grew after an
        empty plan took out eight Llama cells.
        """
        texts = list(texts)
        if not texts:
            return []
        hidden = int(self.model.config.hidden_size)
        records = [self._prepare(text) for text in texts]
        out: list[np.ndarray | None] = [None] * len(records)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        order = sorted(range(len(records)), key=lambda i: len(records[i][0]))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            width = max(len(records[i][0]) for i in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for j, i in enumerate(batch):
                ids = records[i][0]
                input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[j, : len(ids)] = 1
            states = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attn.to(self.model.device),
                output_hidden_states=True,
            ).hidden_states[self.layer]
            # float16 then float32 is not a detour: it is the exact numeric path
            # the parquet cache took, so the clean and intervened activations
            # carry the same precision.
            states = states.to(torch.float16).cpu()
            for j, i in enumerate(batch):
                keep = records[i][1]
                out[i] = (
                    states[j, keep].numpy().astype(np.float32)
                    if keep
                    else np.zeros((0, hidden), dtype=np.float32)
                )
            del states
        return [array for array in out if array is not None]

    def post_token_offsets(self, text: str) -> np.ndarray:
        """[T, 2] character offsets of the tokens `encode` keeps, POST-relative.

        The activation cache stores offsets in the post's own coordinates, not
        the chat template's, and every word alignment in the study
        (`src.probe.word_targets`, `sae_features.aggregate_word_scores`) reads
        them that way. An out-of-domain text has no cache row to read them off,
        so they are recomputed here -- through `_prepare`, so which tokens
        survive is decided in exactly one place.
        """
        full, p0, _ = build_prompt(self.tokenizer, text)
        enc = self.tokenizer(
            full,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )
        _, keep = self._prepare(text)
        offsets = np.array([enc["offset_mapping"][t] for t in keep],
                           dtype=np.int64).reshape(-1, 2)
        return offsets - p0

    def cache_parity(self, text: str, cached: np.ndarray) -> float:
        """Relative L2 between the cache and a fresh encode of the same post.

        The clean logit is read off the cache and the deletion logits off this
        encoder, so a mismatch here would put two different models' numbers in
        one ratio. Reported on every row rather than assumed.
        """
        fresh = self.encode([text], batch_size=1)[0]
        if fresh.shape != cached.shape:
            return float("inf")
        scale = np.linalg.norm(cached)
        if scale <= 0:
            return float("nan")
        return float(np.linalg.norm(fresh - cached) / scale)


def probe_word_deletion_rows(*, backbone, scorer, text, word_scores, cached,
                             config, seed, method, model, variant, fold, pid,
                             ranking_source: str = "own_attribution"):
    """Word-deletion rows for one probe on one post.

    The empty endpoint is the probe's score on an EMPTY token matrix -- its own
    intercept -- matching what `normalized_logit_decay` expects and what the
    arm audit uses, so the two families' decays are the same ratio.

    `word_scores` decides WHICH words are deleted; `scorer` decides what the
    deletion costs. They stay separable -- the split is what let one shared
    external ranking be passed here instead of the probe's own -- but the only
    caller left is `src.stage2.word_deletion`, which always passes the probe's
    own attribution, so `ranking_source` is now always `own_attribution`. It is
    still written on every row: it is the field that would catch a future
    ranking being pooled with these.
    """
    fc = config["faithfulness"]
    k_grid = [float(k) for k in fc["k_grid"]]
    draws = int(fc["random_draws"])
    rng = deterministic_rng(seed, text)

    clean = float(scorer(np.asarray(cached, dtype=np.float64)))
    empty = float(scorer(np.zeros((1, np.asarray(cached).shape[1]),
                                  dtype=np.float64)))
    gap = clean - empty
    common = {
        "method": method, "model": model, "variant": variant, "fold": fold,
        "id": pid,
        "clean_logit": clean, "empty_logit": empty,
        "clean_empty_logit_gap": gap,
        "small_logit_gap": bool(abs(gap) < float(fc["small_logit_gap_warn"])),
        "cache_parity_rel_l2": backbone.cache_parity(text, cached),
        "ranking_source": ranking_source,
        "audit_protocol": "own_word_v2",
        "intervention": "word_delete",
    }

    post_words = words(text)
    scores = np.asarray(word_scores, dtype=float)
    valid = np.flatnonzero(np.isfinite(scores))
    if not len(valid):
        return []
    active = valid[scores[valid] > 0]
    ranked = active[np.argsort(-scores[active], kind="stable")]
    n_active = int(active.size)

    variants, plans = [], []
    rows = []
    for k in k_grid:
        n_selected = min(k_word_count(n_active, k), n_active)
        if not n_selected:
            rows.append({**common, "control": "target", "k_percent": k,
                         "n_selected": 0, "replicate": 0,
                         "intervened_logit": clean,
                         "normalized_decay": normalized_logit_decay(
                             clean, clean, empty)})
            continue
        variants.append(delete_words(text, post_words,
                                     set(ranked[:n_selected].tolist())))
        plans.append(("target", k, n_selected, 0))
        for replicate in range(draws):
            drop = set(rng.choice(valid, size=n_selected, replace=False).tolist())
            variants.append(delete_words(text, post_words, drop))
            plans.append(("random_words", k, n_selected, replicate))

    if variants:
        encoded = backbone.encode(variants)
        for (control, k, n_selected, replicate), vecs in zip(plans, encoded):
            logit = (float(scorer(np.asarray(vecs, dtype=np.float64)))
                     if len(vecs) else empty)
            rows.append({**common, "control": control, "k_percent": k,
                         "n_selected": n_selected, "replicate": replicate,
                         "intervened_logit": logit,
                         "normalized_decay": normalized_logit_decay(
                             clean, logit, empty)})
    _add_aopc(rows, k_grid)
    return rows
