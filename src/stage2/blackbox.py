"""A trained read-out as a black box over TEXT: score a list of strings.

Two wrappers with one interface, used by the input-level attributions that
never read the classifier's internals (src.stage2.occlusion):

    box.score_texts(list[str]) -> np.ndarray of decision logits
    box.clean_logit(text)      -> float
    box.deletion_rows(...)     -> the word-deletion audit rows for one post,
                                  ranked by whatever word scores are passed
    box.release()

`ArmBox` wraps a fine-tuned checkpoint (fine_tune.load_checkpoint and the
same batched scorer the faithfulness audit uses). `ProbeBox` wraps a frozen
probe: the perturbed texts are re-encoded by the frozen backbone
(`src.utils.frozen_backbone.FrozenBackbone`, the corpus cache's own encoder,
at the probe's own layer -- common.probe_layer_of) and read out by the
probe's fitted map -- the two objects the probe's own deletion audit uses.

`broadcast_to_tokens` lifts a per-word score vector to the token sequence
(each aligned token takes its word's score), the shape the plausibility
scorers read.
"""

from __future__ import annotations

import numpy as np

from src.utils.common import ARM_OF, checkpoint_path, probe_layer_of


def broadcast_to_tokens(word_scores: np.ndarray, alignment, n_tokens: int):
    """Sequence-length scores: each aligned token takes its word's score (max
    where a token straddles two words), zero elsewhere."""
    seq = np.zeros(n_tokens, dtype=float)
    seen = np.zeros(n_tokens, dtype=bool)
    for t, w in zip(alignment.token_indices, alignment.word_indices):
        t, w = int(t), int(w)
        value = word_scores[w]
        if not np.isfinite(value):
            continue
        seq[t] = value if not seen[t] else max(seq[t], value)
        seen[t] = True
    return seq


class ArmBox:
    """A fine-tuned checkpoint as a black box, plus its deletion audit."""

    def __init__(self, base: str, model_name: str, cell: int, cfg: dict,
                 batch_size: int):
        from src.baselines import fine_tune as ft
        from src.utils.sae_features import PooledClassifierHookAdapter

        self.checkpoint = checkpoint_path(ARM_OF[base], model_name, cell)
        self.ok = self.checkpoint.exists()
        if not self.ok:
            return
        self.model, self.tok = ft.load_checkpoint(self.checkpoint, model_name)
        self.adapter = PooledClassifierHookAdapter(self.model)
        self.max_len = int(cfg["full_finetune"]["max_len"])
        self.batch_size = batch_size
        self.describe = self.checkpoint.name

    def score_texts(self, texts):
        from src.utils.faithfulness import _score_texts

        return _score_texts(self.model, self.tok, texts, self.max_len,
                            self.batch_size)

    def clean_logit(self, text):
        return float(self.score_texts([text])[0])

    def deletion_rows(self, text, word_scores, *, cfg, seed, name, model_name,
                      cell, pid, ranking_source):
        from src.utils.faithfulness import arm_word_deletion_rows

        return [{"arm": name, "model": model_name, "fold": cell, "id": pid, **r}
                for r in arm_word_deletion_rows(
                    self.adapter, self.tok, text, word_scores, config=cfg,
                    finetune_config=cfg["full_finetune"], seed=seed,
                    ranking_source=ranking_source)]

    def release(self):
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ProbeBox:
    """A frozen probe as a black box: re-encode, then read out."""

    def __init__(self, base: str, model_name: str, cell: int, cfg: dict,
                 batch_size: int, seed: int):
        from src import probe as probing
        from src.utils.frozen_backbone import FrozenBackbone

        layer = probe_layer_of(base, model_name)
        self.ctx = probing.build_context(model_name, layer, cfg, seed)
        self.fitted = probing.FITTERS[base](probing.for_fold(self.ctx, cell))
        self.max_len = int(cfg["full_finetune"]["max_len"])
        self.backbone = FrozenBackbone.load(model_name, layer, self.max_len)
        self.tok = self.backbone.tokenizer
        self.batch_size = batch_size
        self.ok = True
        self.describe = f"frozen {model_name} layer {layer} + {base}"

    def score_texts(self, texts):
        out = np.empty(len(texts), dtype=float)
        encoded = self.backbone.encode(list(texts), batch_size=self.batch_size)
        hidden = int(self.backbone.model.config.hidden_size)
        # A variant with every word deleted has no post token; its score is
        # the probe's read-out of the zero vector -- the same empty endpoint
        # `probe_word_deletion_rows` uses -- rather than a mean over nothing.
        empty = float(self.fitted.score_vectors(np.zeros((1, hidden))))
        for i, vecs in enumerate(encoded):
            out[i] = (float(self.fitted.score_vectors(
                np.asarray(vecs, dtype=np.float64))) if len(vecs) else empty)
        return out

    def clean_logit(self, text):
        return float(self.score_texts([text])[0])

    def deletion_rows(self, text, word_scores, *, cfg, seed, name, model_name,
                      cell, pid, ranking_source):
        from src.utils.frozen_backbone import probe_word_deletion_rows

        return probe_word_deletion_rows(
            backbone=self.backbone, scorer=self.fitted.score_vectors,
            text=text, word_scores=word_scores,
            cached=self.ctx.acts[pid]["vecs"], config=cfg, seed=seed,
            method=name, model=model_name, variant="", fold=cell, pid=pid,
            ranking_source=ranking_source)

    def release(self):
        import torch

        del self.backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SAEBox:
    """A frozen SAE probe as a black box: re-encode at SAE_PROBE_LAYER, push
    the fresh vectors through the pinned dictionary, read out.

    Mirrors ProbeBox with `src.utils.sae_frozen_probe.SAEFrozenReadout` in
    place of a dense `Fitted`; `own_word_scores` is the readout's own token
    decomposition on the cached vectors, aggregated to words.
    """

    def __init__(self, base: str, model_name: str, cell: int, cfg: dict,
                 batch_size: int, seed: int, allow: bool):
        from src.utils.data_prep import load_layer
        from src.utils.frozen_backbone import FrozenBackbone
        from src.utils.sae_frozen_probe import fit_readouts

        readouts, self.posts, layer = fit_readouts(base, model_name, [cell], seed,
                                                   cfg, allow)
        self.readout = readouts[cell]
        self.acts = load_layer(model_name, "lonxplain", layer)
        self.max_len = int(cfg["full_finetune"]["max_len"])
        self.backbone = FrozenBackbone.load(model_name, layer, self.max_len)
        self.tok = self.backbone.tokenizer
        self.batch_size = batch_size
        self.ok = True
        self.describe = f"frozen {model_name} layer {layer} + {base} (SAE)"

    def score_texts(self, texts):
        out = np.empty(len(texts), dtype=float)
        encoded = self.backbone.encode(list(texts), batch_size=self.batch_size)
        hidden = int(self.backbone.model.config.hidden_size)
        empty = float(self.readout.score_vectors(np.zeros((1, hidden))))
        for i, vecs in enumerate(encoded):
            out[i] = (float(self.readout.score_vectors(
                np.asarray(vecs, dtype=np.float64))) if len(vecs) else empty)
        return out

    def clean_logit(self, text):
        return float(self.score_texts([text])[0])

    def own_word_scores(self, text, pid, span, reduction):
        from src.probe import word_targets
        from src.utils.sae_features import aggregate_word_scores

        vecs = np.asarray(self.acts[pid]["vecs"], dtype=np.float64)
        targets = word_targets(np.asarray(self.acts[pid]["offsets"]), text,
                               int(span[0]), int(span[1]))
        tok = self.readout.token_vectors_scores(vecs)
        return aggregate_word_scores(tok, targets.alignment, reduction=reduction)

    def deletion_rows(self, text, word_scores, *, cfg, seed, name, model_name,
                      cell, pid, ranking_source):
        from src.utils.frozen_backbone import probe_word_deletion_rows

        return probe_word_deletion_rows(
            backbone=self.backbone, scorer=self.readout.score_vectors,
            text=text, word_scores=word_scores,
            cached=np.asarray(self.acts[pid]["vecs"], dtype=np.float64),
            config=cfg, seed=seed, method=name, model=model_name, variant="",
            fold=cell, pid=pid, ranking_source=ranking_source)

    def release(self):
        import torch

        del self.backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------ text-only black boxes
#
# The two reference detectors read raw text with their own tokenisers, so
# they have neither the study backbone's alignment nor a residual cache. Their
# occlusion cells are scored on WORDS directly (`word_auprc_from_scores`) and
# audited with a text-level deletion (`text_word_deletion_rows`) whose empty
# endpoint is the model's own score of an empty post.


class MentalBERTBox:
    """The MentalBERT reference classifier as a black box over text."""

    def __init__(self, base: str, model_name: str, cell: int, cfg: dict,
                 batch_size: int):
        from src.baselines import mental_bert
        from src.utils.common import MENTALBERT_ENCODER

        self.checkpoint = checkpoint_path("mentalbert", MENTALBERT_ENCODER, cell)
        self.ok = (self.checkpoint / "model.pt").exists()
        if not self.ok:
            return
        self.model, self.tok, self.max_len = mental_bert.load_checkpoint(self.checkpoint)
        self.cfg = cfg["encoder_finetune"]
        self.batch_size = batch_size
        self.describe = self.checkpoint.name
        self.text_only = True

    def score_texts(self, texts):
        from src.baselines import mental_bert

        return mental_bert.predict(self.model, self.tok, list(texts), self.cfg,
                                   self.max_len, batch_size=self.batch_size)

    def clean_logit(self, text):
        return float(self.score_texts([text])[0])

    def release(self):
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GloVeBox:
    """The GloVe+BiGRU reference classifier as a black box over text."""

    def __init__(self, base: str, model_name: str, cell: int, cfg: dict,
                 batch_size: int):
        from src.baselines import glove_bigru as gb
        from src.utils.common import DEFAULT_MODEL

        self.checkpoint = checkpoint_path("glove_bigru", DEFAULT_MODEL, cell)
        self.ok = (self.checkpoint / "model.pt").exists()
        if not self.ok:
            return
        self.model = gb.load_checkpoint(self.checkpoint, cfg)
        self.cfg = cfg
        self.batch_size = batch_size
        self.describe = self.checkpoint.name
        self.text_only = True

    def score_texts(self, texts):
        from src.baselines import glove_bigru as gb

        return gb.predict(self.model, list(texts), self.cfg,
                          batch_size=self.batch_size)

    def clean_logit(self, text):
        return float(self.score_texts([text])[0])

    def release(self):
        del self.model


def word_auprc_from_scores(text: str, scores: np.ndarray, span) -> dict:
    """Word AUPRC of a per-word score vector against the expert span, on the
    contract `src.utils.dla._word_plausibility` applies: scored words are
    those with a normalised form and a finite score; gold is character
    overlap with [span_start, span_end); NaN when single-class."""
    from sklearn.metrics import average_precision_score

    from src.utils.data_prep import words

    post_words = words(text)
    scored = [i for i, w in enumerate(post_words)
              if w["norm"] and i < len(scores) and np.isfinite(scores[i])]
    gold = np.array([post_words[i]["start"] < int(span[1])
                     and post_words[i]["end"] > int(span[0]) for i in scored],
                    dtype=bool)
    n_gold = int(gold.sum())
    out = {"n_words_scored": len(scored), "n_gold_words": n_gold,
           "word_gold_base_rate": (n_gold / len(scored)) if scored else float("nan"),
           "word_aggregation": "word"}
    if not scored or len(np.unique(gold)) < 2:
        out["word_auprc"] = float("nan")
    else:
        out["word_auprc"] = float(average_precision_score(
            gold.astype(int), np.asarray(scores, dtype=float)[scored]))
    return out


def text_word_deletion_rows(score_texts, text: str, word_scores, *, config: dict,
                            seed: int, ranking_source: str) -> list[dict]:
    """`src.utils.faithfulness.arm_word_deletion_rows` for a black box over
    text: the same grid, count-matched random control and normalised decay;
    the empty endpoint is the model's score of an empty post."""
    from src.utils.data_prep import words
    from src.utils.faithfulness import (_add_aopc, delete_words, deterministic_rng,
                                        k_word_count, normalized_logit_decay)

    fc = config["faithfulness"]
    k_grid = [float(k) for k in fc["k_grid"]]
    draws = int(fc["random_draws"])
    rng = deterministic_rng(seed, text)
    post_words = words(text)
    clean, empty = (float(v) for v in score_texts([text, ""]))
    gap = clean - empty
    common = {"clean_logit": clean, "empty_logit": empty,
              "clean_empty_logit_gap": gap,
              "small_logit_gap": bool(abs(gap) < float(fc.get("small_logit_gap_warn", 0.1))),
              "ranking_source": ranking_source, "audit_protocol": "own_word_v2"}
    scores = np.asarray(word_scores, dtype=float)
    valid = np.flatnonzero(np.isfinite(scores))
    if not len(valid):
        return []
    active = valid[scores[valid] > 0]
    ranked = active[np.argsort(-scores[active], kind="stable")]
    common["n_active_words"] = int(active.size)
    common["n_valid_words"] = int(valid.size)
    rows, variants, plans = [], [], []
    for k in k_grid:
        n_sel = min(k_word_count(int(active.size), k), int(active.size))
        if not n_sel:
            rows.append({**common, "intervention": "word_delete", "control": "target",
                         "k_percent": k, "n_selected": 0, "replicate": 0,
                         "intervened_logit": clean,
                         "normalized_decay": normalized_logit_decay(clean, clean, empty)})
            continue
        variants.append(delete_words(text, post_words, set(ranked[:n_sel].tolist())))
        plans.append(("target", k, n_sel, 0))
        for r in range(draws):
            drop = set(rng.choice(valid, size=n_sel, replace=False).tolist())
            variants.append(delete_words(text, post_words, drop))
            plans.append(("random_words", k, n_sel, r))
    if variants:
        logits = np.asarray(score_texts(variants), dtype=float)
        for (control, k, n_sel, r), logit in zip(plans, logits):
            rows.append({**common, "intervention": "word_delete", "control": control,
                         "k_percent": k, "n_selected": n_sel, "replicate": r,
                         "intervened_logit": float(logit),
                         "normalized_decay": normalized_logit_decay(clean, float(logit), empty)})
    _add_aopc(rows, k_grid)
    return rows
