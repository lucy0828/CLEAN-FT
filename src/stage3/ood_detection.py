"""Stage 3, step 5: the Stage 1 detectors on the conversation phases, scored
against the two clinicians -> results/ood_transfer.csv.

How to run. The GPU pass re-encodes conversation text through a study backbone
or a retained checkpoint (nothing comes off the LonXplain activation cache);
the collect pass is CPU and reads the rows the GPU pass wrote:

    uv run python -m src.stage3.ood_detection --allow-model-process --devices 1
    uv run python -m src.stage3.ood_detection --model llama --allow-model-process --devices 2
    uv run python -m src.stage3.ood_detection --method attention_probe full_ft \
                                              --allow-model-process
    uv run python -m src.stage3.ood_detection --limit 8 --allow-model-process   # smoke
    uv run python -m src.stage3.ood_detection --collect                          # both models

Inputs:
    results/stage3/screened_phases_{corpus}.parquet  which phases the screen read
    data/conversation/{corpus}/dialogue.csv          the cleaned dialogue
    results/stage3/annotations.csv                   both clinicians' labels and
                                                     spans (src.stage3.annotator_labels)
    results/detection.csv, dla_auprc.csv,            the LonXplain numbers the
    word_deletion.csv                                OOD ones are set beside

Outputs:
    results/raw/stage3/ood_{method}_{model}.parquet        every screened phase:
        score, transferred threshold, flag, and both annotators' labels
    results/raw/stage3/oodwords_{method}_{model}.parquet   every screened phase,
        one row per word: the method's own attribution and its rank
    results/raw/stage3/ooddelete_{method}_{model}.parquet  every ANNOTATED phase:
        the word-deletion rows (src.utils.faithfulness / frozen_backbone)
    results/ood_transfer.csv                               the published table:
        AUROC, word AUPRC and AOPC on LonXplain and on the conversations, per
        method, backbone and annotator view
    results/stage3/OOD.md                                  the same numbers as
        readable tables, under the annotators' agreement -- rendered from the
        CSVs on every --collect, never edited by hand

THE SAME THREE NUMBERS AS STAGES 1 AND 2, ON NEW TEXT
-----------------------------------------------------
Every method is carried over WITHOUT retraining and without refitting its
threshold (`DEPLOY_FOLD` below), and the three Stage 1/2 metrics are read on
the conversations exactly as they were read on LonXplain:

    auroc       detection: the method's phase score against the clinician's
                label (src.utils.common.detection_metrics's AUROC)
    word_auprc  plausibility: the method's OWN word attribution ranked
                against the clinician's quoted spans (src.probe.post_word_auprc,
                same alignment, same sub-word reduction)
    aopc        faithfulness: delete the method's own top-ranked words, re-score,
                margin over a count-matched random control
                (src.stage2.word_deletion._margins, same k grid, same draws)

WHO IS THE GROUND TRUTH: THREE VIEWS, NOTHING ADJUDICATED
---------------------------------------------------------
Two clinicians labelled the 34 phases the screen put in front of them
(results/stage3/annotations.csv keeps both as columns). Every metric is
therefore published three times:

    annotator1, annotator2   that clinician's labels and spans alone
    pooled                   both readings kept: a phase the two clinicians
                             disagree on appears TWICE, once under each
                             label (and once under each span set), a phase
                             they agree on appears once

The pooled view is the "duplicate disputed instances" rule -- applied at
evaluation, because nothing in Stage 3 trains on conversation data. It
weights a disputed phase by both readings instead of picking one.

THE NEGATIVES
-------------
The clinicians read only the phases the screen flagged. A detection AUROC
needs negatives, and 32 of 34 phases carry annotator 1's tick, so the
annotated set alone cannot rank a detector. The published `auroc` is
therefore read over EVERY screened phase: the clinician's label where there is
one, and 0 for the phases the screen did not flag -- a label the screening
model gave, not a clinician, which the table says in `n_neg_screen`. The AUROC
over the annotated phases alone is published beside it as `auroc_annotated`,
with its own (small) counts, so a reader can see both.

ONE DEPLOYED CELL PER METHOD: FOLD 0
------------------------------------
Each method is deployed as its fold-0 cell -- a probe fitted on the 9/10 of
LonXplain outside fold 0, an arm loaded from the fold-0 checkpoint, which is
the cell `src.stage1.train` always retains. No conversation phase is in any
LonXplain fold, so nothing is in-fold or out-of-fold: it is out-of-corpus.
Thresholds transfer unchanged (a probe's refitted exactly as
`src.stage1.train.training_threshold` computed it, an arm's read off its
recipe sidecar); they decide only the `flagged` column, which no published
metric reads.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.stage3.ood_corpus import SPEAKER, load_phases
from src.utils.common import (ARM_OF, BACKBONE_FREE, CELLS_FROM,
                              DEFAULT_MODEL, DENSE_PROBE_LAYER,
                              DENSE_PROBES, EXPLAINER_BASE, RESULTS,
                              ROOT, SAE_PROBE_LAYER, STAGE3, add_device_args,
                              add_method_args,
                              artifact_path, ci, group_of, load_config,
                              mean_ci, metric_distribution, min_free_gb_for,
                              pin_visible_devices, record_decision,
                              resolve_model, selected_methods, shared_draws)

# Every method whose Stage 1 output can be re-run on new text: the dense
# frozen probes, the mean-pooled SAE probe (through src.utils.sae_frozen_probe,
# which re-encodes at SAE_PROBE_LAYER and pushes the fresh vectors through the
# pinned dictionary), and the fine-tuning arms with a retained fold-0
# checkpoint. glove_bigru and mentalbert retain no checkpoint and prompting
# has no weights, so none of those is deployed.
DENSE = tuple(m for m in DENSE_PROBES
              if m not in ("shrunk_attention_probe", "final_linear_probe",
                           "final_attention_probe"))
SAE = ("sae_probe",)
ARMS = tuple(m for m in ARM_OF if m not in CELLS_FROM)
# The two reference detectors, deployed from their retained fold-0 checkpoint
# as text-in black boxes with their own (mean-pooled, linear-head) DLA
# (src.stage2.reference_dla); filed under the backbone label they run beside.
TEXT = ("mentalbert", "glove_bigru")
METHODS = DENSE + SAE + ARMS + TEXT

# The cells the OOD table publishes by default: the two final-layer
# fine-tunes, the mid-layer probe-initialised fine-tune, and the two
# attention-pooled probes. `midlayer_ft` and `sae_probe` were deployed on
# 2026-09-07 (their raw rows are on disk) and then dropped from the published
# set at the user's request; pass them with --method to collect them.
STUDY_METHODS = ("full_ft", "attn_ft", "midlayer_attn_ft", "attn_lora_ft",
                 "attn_lp_ft", "midlayer_attn_lp_ft",
                 "attention_mlp_probe", "attention_probe", "attn_span_ft",
                 "mentalbert", "glove_bigru")

DEPLOY_FOLD = 0
UNIT = "conversation_phase"
SEED = 0
VIEWS = ("annotator1", "annotator2", "pooled")
MODELS = ("gemma-2-2b", "llama31-8b")

ANNOTATIONS_CSV = STAGE3 / "annotations.csv"
OUT_CSV = RESULTS / "ood_transfer.csv"

PHASE_ROW_COLUMNS = ["method", "group", "model", "fold", "unit",
                     "scored_speaker", "corpus", "phase_id", "conversation_id",
                     "type", "nickname", "date", "n_user_messages",
                     "n_user_words", "n_tokens", "truncated",
                     "in_clinician_script", "score", "threshold",
                     "threshold_source", "flagged"]

WORD_ROW_COLUMNS = ["method", "model", "phase_id", "word_index", "char_start",
                    "char_end", "word", "turn_order", "score", "rank"]


# ================================================================ annotations


def load_annotations(required: bool = True) -> pd.DataFrame | None:
    if not ANNOTATIONS_CSV.exists():
        if required:
            raise SystemExit(
                f"missing {ANNOTATIONS_CSV.relative_to(ROOT)} -- build it with "
                "`uv run python -m src.stage3.annotator_labels --annotator ... "
                "--annotator ...`")
        return None
    table = pd.read_csv(ANNOTATIONS_CSV)
    for k in (1, 2):
        table[f"annotator{k}_label"] = table[f"annotator{k}_label"].astype(int)
        table[f"annotator{k}_span_chars"] = table[
            f"annotator{k}_span_chars"].fillna("[]")
    return table


def supervision(annotations: pd.DataFrame, view: str) -> pd.DataFrame:
    """One row per (phase, reading) the view evaluates against.

    `annotator1` / `annotator2`: that clinician's label and spans, one row per
    phase. `pooled`: both clinicians' rows, with a phase whose two readings
    are identical (same label and same span set) kept once -- so an agreed
    phase counts once and a disputed one twice, one row per reading.
    """
    parts = []
    for k in (1, 2):
        if view != "pooled" and view != f"annotator{k}":
            continue
        parts.append(pd.DataFrame({
            "phase_id": annotations["phase_id"],
            "reader": f"annotator{k}",
            "label": annotations[f"annotator{k}_label"].astype(int),
            "span_chars": annotations[f"annotator{k}_span_chars"],
        }))
    table = pd.concat(parts, ignore_index=True)
    if view == "pooled":
        table = table.drop_duplicates(subset=["phase_id", "label", "span_chars"])
    return table.reset_index(drop=True)


# ============================================================== word scoring


def word_scores_from_tokens(token_scores, offsets, text: str,
                            reduction: str) -> np.ndarray:
    """One score per word of `text`, NaN where no token reached the word.

    The same two steps every other word-level readout in the study takes --
    `src.probe.word_targets` for the character-overlap alignment, then
    `sae_features.aggregate_word_scores` under the pinned reduction -- so a
    word ranks here exactly as it ranks in Stage 2.
    """
    from src.probe import word_targets
    from src.utils.sae_features import aggregate_word_scores

    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    if not len(offsets) or not len(token_scores):
        return np.full(len(_words(text)), np.nan)
    targets = word_targets(offsets, text, -1, -1)
    return aggregate_word_scores(np.asarray(token_scores, dtype=np.float64),
                                 targets.alignment, reduction=reduction)


def _words(text: str):
    from src.utils.data_prep import words

    return words(text)


def turn_of(bounds, char_start: int) -> int:
    """The `order` of the older-adult message a character position falls in."""
    for start, end, order in bounds:
        if start <= char_start < end:
            return order
    return -1


def word_rows(method: str, model: str, phase: dict, word_scores,
              bounds) -> list[dict]:
    text = phase["text"]
    post_words = _words(text)
    scores = np.asarray(word_scores, dtype=float)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(len(scores))
    return [{
        "method": method, "model": model, "phase_id": phase["phase_id"],
        "word_index": i, "char_start": int(w["start"]),
        "char_end": int(w["end"]), "word": text[w["start"]:w["end"]],
        "turn_order": turn_of(bounds, int(w["start"])),
        "score": float(scores[i]), "rank": int(rank[i]),
    } for i, w in enumerate(post_words)]


def phase_word_auprc(text: str, word_scores: np.ndarray,
                     span_chars: list[list[int]]) -> tuple[float, int, int]:
    """(word AUPRC, n scored words, n gold words) for one phase.

    The contract is `src.probe.post_word_auprc`'s, decision for decision: a
    word is scored when it has a normalised form and at least one token
    reached it (a finite score); it is gold when its characters overlap any
    of the clinician's located quotes; the AUPRC is sklearn's average
    precision over the scored words and is NaN when they are single-class.
    The one difference is that a clinician may quote several places in one
    phase, so gold is overlap with ANY range rather than with one span.
    """
    post_words = _words(text)
    scores = np.asarray(word_scores, dtype=float)
    scored = [i for i, w in enumerate(post_words)
              if w["norm"] and i < len(scores) and np.isfinite(scores[i])]
    if not scored:
        return float("nan"), 0, 0
    gold = np.array([any(post_words[i]["start"] < end
                         and post_words[i]["end"] > start
                         for start, end in span_chars) for i in scored],
                    dtype=bool)
    n_gold = int(gold.sum())
    if len(np.unique(gold)) < 2:
        return float("nan"), len(scored), n_gold
    return (float(average_precision_score(gold.astype(int), scores[scored])),
            len(scored), n_gold)


# ==================================================================== methods


def deploy_dense_probes(methods, model: str, phases, turns_of, deletion_ids,
                        cfg, *, chunk: int, batch_size: int) -> dict:
    """{method: result} for the frozen probes, fitted at fold 0 and read on
    fresh token vectors at DENSE_PROBE_LAYER."""
    from src import probe as probing
    from src.stage1.train import training_threshold
    from src.utils.frozen_backbone import FrozenBackbone, probe_word_deletion_rows

    layer = DENSE_PROBE_LAYER[model]
    max_len = int(cfg["full_finetune"]["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]
    print(f"dense probes: {model} layer {layer}, fitted on fold "
          f"{DEPLOY_FOLD}'s training posts", flush=True)
    ctx = probing.build_context(model, layer, cfg, SEED)
    fold_ctx = probing.for_fold(ctx, DEPLOY_FOLD)
    train_idx = np.asarray(fold_ctx.train_idx, dtype=int)

    fitted_of, thresholds = {}, {}
    for method in methods:
        fitted = probing.FITTERS[method](fold_ctx)
        if fitted.score_vectors is None or fitted.token_vectors_scores is None:
            print(f"  {method}: {fitted.token_source} has no readout for "
                  "fresh token vectors -- SKIPPED")
            continue
        fitted_of[method] = fitted
        thresholds[method] = training_threshold(fitted.post_scores(train_idx),
                                                ctx.y[train_idx])
        print(f"  {method}: threshold {thresholds[method]:.4f}", flush=True)
    if not fitted_of:
        return {}

    records = phases.to_dict("records")
    out = {m: {"score": [], "words": [], "word_rows": [], "delete_rows": []}
           for m in fitted_of}
    kept, cut = [], []
    backbone = FrozenBackbone.load(model, layer, max_len)
    try:
        for start in range(0, len(records), chunk):
            block = records[start:start + chunk]
            texts = [p["text"] for p in block]
            offsets = [backbone.post_token_offsets(t) for t in texts]
            vectors = backbone.encode(texts, batch_size=batch_size)
            assert len(vectors) == len(block), "backbone returned a short batch"
            for phase, off, vecs in zip(block, offsets, vectors):
                text = phase["text"]
                kept.append(int(len(vecs)))
                cut.append(bool(len(off) and len(_words(text))
                                and int(off[-1][1]) < len(text.rstrip())))
                for method, fitted in fitted_of.items():
                    if not len(vecs):
                        out[method]["score"].append(float("nan"))
                        ws = np.full(len(_words(text)), np.nan)
                    else:
                        V = np.asarray(vecs, dtype=np.float64)
                        out[method]["score"].append(
                            float(fitted.score_vectors(V)))
                        ws = word_scores_from_tokens(
                            fitted.token_vectors_scores(V), off, text,
                            reduction)
                    out[method]["words"].append(ws)
                    out[method]["word_rows"] += word_rows(
                        method, model, phase, ws, turns_of[phase["phase_id"]])
                    if phase["phase_id"] in deletion_ids and len(vecs) \
                            and np.isfinite(ws).any():
                        out[method]["delete_rows"] += probe_word_deletion_rows(
                            backbone=backbone, scorer=fitted.score_vectors,
                            text=text, word_scores=ws,
                            cached=np.asarray(vecs, dtype=np.float64),
                            config=cfg, seed=SEED, method=method, model=model,
                            variant="", fold=DEPLOY_FOLD,
                            pid=phase["phase_id"])
            print(f"    scored {min(start + chunk, len(records))}/"
                  f"{len(records)} phases", flush=True)
    finally:
        _release(backbone.model)

    return {m: {"scores": np.asarray(out[m]["score"], dtype=float),
                "word_rows": out[m]["word_rows"],
                "delete_rows": out[m]["delete_rows"],
                "n_tokens": kept, "truncated": cut,
                "threshold": thresholds[m],
                "threshold_source":
                    f"refit of the fold-{DEPLOY_FOLD} training-fold cut"}
            for m in fitted_of}


def deploy_sae_probe(model: str, phases, turns_of, deletion_ids, cfg, *,
                     allow: bool, chunk: int, batch_size: int) -> dict:
    """The mean-pooled SAE probe, fitted at fold 0, read through the pinned
    dictionary on fresh token vectors at SAE_PROBE_LAYER.

    A second backbone pass at a second layer: the dense probes read the 0.65
    working layer and the dictionary lives one block below it on Gemma (cache
    18 against 17) and at the same index on Llama (24). The readout is
    `src.utils.sae_frozen_probe.SAEFrozenReadout` -- the composition word
    deletion already audits this probe through -- so the score, the word
    attribution and the deletion decays all go through one dictionary.
    """
    from src.probe import sae_probe
    from src.stage1.train import training_threshold
    from src.utils.frozen_backbone import FrozenBackbone, probe_word_deletion_rows
    from src.utils.sae_frozen_probe import fit_readouts

    layer = SAE_PROBE_LAYER[model]
    max_len = int(cfg["full_finetune"]["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]
    print(f"sae_probe: {model} cached layer {layer}, fitted on fold "
          f"{DEPLOY_FOLD}'s training posts", flush=True)
    readouts, posts, _ = fit_readouts("sae_probe", model, [DEPLOY_FOLD], SEED,
                                      cfg, allow)
    readout = readouts[DEPLOY_FOLD]
    # The transferred threshold: the training-fold cut on the same pooled
    # latents the probe was fitted on (served from the corpus latent cache).
    latents, _ = sae_probe.encode_posts_to_latents(
        None, model, "lonxplain", layer, post_ids=posts["id"].tolist(),
        pool="mean")
    y = (posts["label"] == "present").to_numpy().astype(int)
    train = np.flatnonzero(posts["fold"].to_numpy() != DEPLOY_FOLD)
    threshold = training_threshold(
        sae_probe.score_sae_probe(readout.probe, latents.X[train]), y[train])
    print(f"  sae_probe: threshold {threshold:.4f}", flush=True)

    records = phases.to_dict("records")
    scores, rows_w, rows_d, kept, cut = [], [], [], [], []
    backbone = FrozenBackbone.load(model, layer, max_len)
    try:
        for start in range(0, len(records), chunk):
            block = records[start:start + chunk]
            texts = [p["text"] for p in block]
            offsets = [backbone.post_token_offsets(t) for t in texts]
            vectors = backbone.encode(texts, batch_size=batch_size)
            for phase, off, vecs in zip(block, offsets, vectors):
                text = phase["text"]
                kept.append(int(len(vecs)))
                cut.append(bool(len(off) and len(_words(text))
                                and int(off[-1][1]) < len(text.rstrip())))
                if not len(vecs):
                    scores.append(float("nan"))
                    ws = np.full(len(_words(text)), np.nan)
                else:
                    scores.append(float(readout.score_vectors(vecs)))
                    ws = word_scores_from_tokens(
                        readout.token_vectors_scores(vecs), off, text,
                        reduction)
                rows_w += word_rows("sae_probe", model, phase, ws,
                                    turns_of[phase["phase_id"]])
                if phase["phase_id"] in deletion_ids and len(vecs) \
                        and np.isfinite(ws).any():
                    rows_d += probe_word_deletion_rows(
                        backbone=backbone, scorer=readout.score_vectors,
                        text=text, word_scores=ws,
                        cached=np.asarray(vecs, dtype=np.float64),
                        config=cfg, seed=SEED, method="sae_probe",
                        model=model, variant="", fold=DEPLOY_FOLD,
                        pid=phase["phase_id"])
            print(f"    scored {min(start + chunk, len(records))}/"
                  f"{len(records)} phases", flush=True)
    finally:
        _release(backbone.model)
    return {"sae_probe": {
        "scores": np.asarray(scores, dtype=float), "word_rows": rows_w,
        "delete_rows": rows_d, "n_tokens": kept, "truncated": cut,
        "threshold": threshold,
        "threshold_source": f"refit of the fold-{DEPLOY_FOLD} training-fold cut",
    }}


def deploy_arms(methods, model: str, phases, turns_of, deletion_ids, cfg, *,
                batch_size: int) -> dict:
    """Each arm's fold-0 checkpoint: its score per phase, its word terms, and
    its word-deletion rows on the annotated phases."""
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.common import checkpoint_path
    from src.utils.faithfulness import arm_word_deletion_rows
    from src.utils.sae_features import (PooledClassifierHookAdapter,
                                        aggregate_word_scores, align_post_tokens)

    ftcfg = cfg["full_finetune"]
    max_len = int(ftcfg["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]
    records = phases.to_dict("records")
    texts = [p["text"] for p in records]

    out = {}
    for method in methods:
        arm = ARM_OF[method]
        ck = checkpoint_path(arm, model, DEPLOY_FOLD)
        sidecar = artifact_path("train", arm, model, fold=DEPLOY_FOLD,
                                ext="json")
        if not ck.exists():
            print(f"  {method}: no fold-{DEPLOY_FOLD} checkpoint at "
                  f"{ck.name} -- SKIPPED. Retain it with `src.stage1.train "
                  f"--method {method} --folds {DEPLOY_FOLD} --keep-checkpoint`;"
                  " this never trains.")
            continue
        if not sidecar.exists():
            print(f"  {method}: {sidecar.name} is missing, so the threshold it "
                  "fitted on its training folds cannot be read -- SKIPPED "
                  "rather than re-cut on conversation data.")
            continue
        with open(sidecar) as fh:
            threshold = float(json.load(fh)["threshold"])
        print(f"  {method}: fold-{DEPLOY_FOLD} checkpoint, threshold "
              f"{threshold:.4f}", flush=True)

        model_obj, tok = ft.load_checkpoint(ck, model)
        try:
            device = next(model_obj.parameters()).device
            adapter = PooledClassifierHookAdapter(model_obj)
            scores = np.asarray(ft.predict(model_obj, tok, texts, ftcfg,
                                           batch_size=batch_size), dtype=float)
            print(f"    scored {len(texts)} phases; attributing their words",
                  flush=True)
            rows_w, rows_d, kept, cut = [], [], [], []
            for i, phase in enumerate(records):
                text = phase["text"]
                alignment = align_post_tokens(tok, text, max_len=max_len)
                kept.append(int(len(np.unique(alignment.token_indices))))
                cut.append(bool(alignment.truncated))
                ids, attn, pm = ft.build_inputs(tok, [text], max_len)
                with torch.no_grad():
                    token = ft.token_scores(model_obj, ids.to(device),
                                            attn.to(device), pm.to(device))
                ws = aggregate_word_scores(token, alignment, reduction=reduction)
                rows_w += word_rows(method, model, phase, ws,
                                    turns_of[phase["phase_id"]])
                if phase["phase_id"] in deletion_ids and np.isfinite(ws).any():
                    for r in arm_word_deletion_rows(
                            adapter, tok, text, ws, config=cfg,
                            finetune_config=ftcfg, seed=SEED):
                        rows_d.append({"method": method, "model": model,
                                       "variant": "", "fold": DEPLOY_FOLD,
                                       "id": phase["phase_id"], **r})
                if (i + 1) % 100 == 0:
                    print(f"    attributed {i + 1}/{len(records)}", flush=True)
        finally:
            _release(model_obj)
        out[method] = {"scores": scores, "word_rows": rows_w,
                       "delete_rows": rows_d, "n_tokens": kept,
                       "truncated": cut, "threshold": threshold,
                       "threshold_source":
                           f"train_{arm}_fold{DEPLOY_FOLD}.json"}
    return out


def _text_threshold(method: str) -> tuple[float, str]:
    """The fold-0 decision threshold of a reference detector: GloVe's from
    its training sidecar; MentalBERT's recovered from its fold-0 held-out
    decisions (the smallest score it called present)."""
    if method == "glove_bigru":
        p = artifact_path("train", "glove_bigru", DEFAULT_MODEL,
                          fold=DEPLOY_FOLD, ext="json")
        with open(p) as fh:
            return float(json.load(fh)["threshold"]), p.name
    from src.utils.common import MENTALBERT_ENCODER

    p = artifact_path("detect", "mentalbert", MENTALBERT_ENCODER)
    d = pd.read_parquet(p)
    d = d[d["fold"] == DEPLOY_FOLD]
    pos = d[d["post_pred"] == "present"]["post_score"]
    neg = d[d["post_pred"] != "present"]["post_score"]
    thr = float(pos.min()) if len(pos) else float(neg.max())
    return thr, f"{p.name} fold-{DEPLOY_FOLD} decisions"


def deploy_text_models(methods, model: str, phases, turns_of, deletion_ids,
                       cfg, *, batch_size: int) -> dict:
    """The two references from their fold-0 checkpoints: score per phase,
    their own mean-pooled DLA per word, and word deletion on text."""
    from src.stage2.blackbox import text_word_deletion_rows
    from src.stage2.reference_dla import RANKING_SOURCE, open_box, word_terms

    records = phases.to_dict("records")
    texts = [p["text"] for p in records]
    out = {}
    for method in methods:
        box = open_box(method, DEPLOY_FOLD, cfg, batch_size)
        if not box.ok:
            print(f"  {method}: no fold-{DEPLOY_FOLD} checkpoint -- SKIPPED")
            continue
        threshold, source = _text_threshold(method)
        print(f"  {method}: fold-{DEPLOY_FOLD} checkpoint, threshold "
              f"{threshold:.4f} ({source})", flush=True)
        try:
            scores = np.asarray(box.score_texts(texts), dtype=float)
            print(f"    scored {len(texts)} phases; attributing their words",
                  flush=True)
            rows_w, rows_d, kept, cut = [], [], [], []
            for i, phase in enumerate(records):
                text = phase["text"]
                ws, _clean, _err, n_tok, truncated = word_terms(box, method, text, cfg)
                kept.append(int(n_tok))
                cut.append(bool(truncated))
                rows_w += word_rows(method, model, phase, ws,
                                    turns_of[phase["phase_id"]])
                if phase["phase_id"] in deletion_ids and np.isfinite(ws).any():
                    for r in text_word_deletion_rows(
                            box.score_texts, text, ws, config=cfg, seed=SEED,
                            ranking_source=RANKING_SOURCE):
                        rows_d.append({"method": method, "model": model,
                                       "variant": "", "fold": DEPLOY_FOLD,
                                       "id": phase["phase_id"], **r})
                if (i + 1) % 100 == 0:
                    print(f"    attributed {i + 1}/{len(records)}", flush=True)
        finally:
            box.release()
        out[method] = {"scores": scores, "word_rows": rows_w,
                       "delete_rows": rows_d, "n_tokens": kept,
                       "truncated": cut, "threshold": threshold,
                       "threshold_source": source}
    return out


def _release(module) -> None:
    import torch

    del module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================== assembly


def phase_rows(method: str, model: str, phases, result,
               annotations: pd.DataFrame | None) -> pd.DataFrame:
    frame = phases.drop(columns=["text"]).copy()
    frame["method"] = method
    frame["group"] = group_of(method)
    frame["model"] = model
    frame["fold"] = DEPLOY_FOLD
    frame["unit"] = UNIT
    frame["scored_speaker"] = SPEAKER
    frame["n_tokens"] = np.asarray(result["n_tokens"], dtype=int)
    frame["truncated"] = np.asarray(result["truncated"], dtype=bool)
    frame["score"] = result["scores"]
    frame["threshold"] = float(result["threshold"])
    frame["threshold_source"] = result["threshold_source"]
    frame["flagged"] = result["scores"] >= float(result["threshold"])
    frame = frame[PHASE_ROW_COLUMNS]
    if annotations is not None:
        labels = annotations[["phase_id", "annotator1_label",
                              "annotator2_label"]]
        frame = frame.merge(labels, on="phase_id", how="left")
    return frame


def phase_path(method: str, model: str, smoke: bool = False):
    return artifact_path("ood", method, model, smoke=smoke)


def word_path(method: str, model: str, smoke: bool = False):
    return artifact_path("oodwords", method, model, smoke=smoke)


def delete_path(method: str, model: str, smoke: bool = False):
    return artifact_path("ooddelete", method, model, smoke=smoke)


# =================================================================== metrics


def _lonxplain_tables() -> dict:
    """The three published Stage 1/2 tables, keyed (method, model)."""
    import re

    out = {}
    det = pd.read_csv(RESULTS / "detection.csv")
    out["auroc"] = {(r.method, r.model): (r.auroc, r.auroc_lo, r.auroc_hi,
                                          int(r.test_post))
                    for r in det.itertuples()}
    dla = pd.read_csv(RESULTS / "dla_auprc.csv")
    dla["variant"] = dla["variant"].fillna("")
    pinned = ~dla["variant"].str.contains(r"_word(?:max|mean|sum)$", regex=True)
    out["word_auprc"] = {(r.method, r.model): (r.word_auprc, r.word_auprc_lo,
                                               r.word_auprc_hi, int(r.n_posts))
                         for r in dla[pinned].itertuples()}
    wd = pd.read_csv(RESULTS / "word_deletion.csv")
    out["aopc"] = {(r.method, r.model): (r.margin, r.margin_lo, r.margin_hi,
                                         int(r.n_posts))
                   for r in wd.itertuples()}
    # the backbone-free references publish detection under their own model
    # label; their OOD rows are filed under the study default backbone
    for table in out.values():
        for (m, mod), v in list(table.items()):
            if m in BACKBONE_FREE and (m, DEFAULT_MODEL) not in table:
                table[(m, DEFAULT_MODEL)] = v
    return out


def _auroc_rows(scores_of: dict, screened: pd.DataFrame, sup: pd.DataFrame,
                annotated_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """(y, s) for one view: the supervision rows, plus the screen-negative
    phases as 0 unless `annotated_only`."""
    y = sup["label"].to_numpy(dtype=int).tolist()
    s = [scores_of[pid] for pid in sup["phase_id"]]
    if not annotated_only:
        rest = screened[~screened["phase_id"].isin(set(sup["phase_id"]))]
        y += [0] * len(rest)
        s += [scores_of[pid] for pid in rest["phase_id"]]
    return np.asarray(y, dtype=int), np.asarray(s, dtype=float)


def _auroc(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    value = float(roc_auc_score(y, s))
    dist = metric_distribution(roc_auc_score, y, s, shared_draws(len(y)))
    lo, hi = ci(dist)
    return value, lo, hi


def collect_metrics(model: str, methods, phases, annotations,
                    smoke: bool = False) -> pd.DataFrame:
    """One long table: method x view x metric, OOD beside LonXplain."""
    from src.stage2.word_deletion import _margins

    lonx = _lonxplain_tables()
    text_of = phases.set_index("phase_id")["text"].to_dict()
    rows = []
    for method in methods:
        p_path, w_path, d_path = (phase_path(method, model, smoke),
                                  word_path(method, model, smoke),
                                  delete_path(method, model, smoke))
        if not p_path.exists():
            print(f"  {method}: {p_path.name} not on disk -- run the GPU pass")
            continue
        scored = pd.read_parquet(p_path)
        screened = scored[["phase_id", "score"]]
        scores_of = dict(zip(screened["phase_id"], screened["score"]))
        words_by_phase = {
            pid: g.sort_values("word_index")["score"].to_numpy(dtype=float)
            for pid, g in pd.read_parquet(w_path).groupby("phase_id")
        } if w_path.exists() else {}
        deletion = pd.read_parquet(d_path) if d_path.exists() else None

        for view in VIEWS:
            sup = supervision(annotations, view)
            positives = sup[sup["label"] == 1]
            base = {"method": method, "group": group_of(method),
                    "model": model, "annotator": view}

            # -- detection AUROC, over the screened corpus and annotated-only
            for metric, annotated_only in (("auroc", False),
                                           ("auroc_annotated", True)):
                y, s = _auroc_rows(scores_of, screened, sup, annotated_only)
                value, lo, hi = _auroc(y, s)
                lx = lonx["auroc"].get((method, model), (np.nan,) * 3 + (0,))
                rows.append({**base, "metric": metric,
                             "lonxplain": lx[0], "lonxplain_lo": lx[1],
                             "lonxplain_hi": lx[2], "n_lonxplain": lx[3],
                             "ood": value, "ood_lo": lo, "ood_hi": hi,
                             "n_ood": int(len(y)), "n_ood_pos": int(y.sum()),
                             "n_ood_neg_annotated": int((sup["label"] == 0).sum()),
                             "n_ood_neg_screen": int(len(y) - len(sup)),
                             "chance_ood": 0.5})

            # -- plausibility: word AUPRC against the clinician's quotes
            values, n_gold, n_scored = [], [], []
            for r in positives.itertuples():
                spans = json.loads(r.span_chars)
                if not spans or r.phase_id not in words_by_phase:
                    continue
                ap, ns, ng = phase_word_auprc(text_of[r.phase_id],
                                              words_by_phase[r.phase_id], spans)
                if np.isfinite(ap):
                    values.append(ap)
                    n_gold.append(ng)
                    n_scored.append(ns)
            value, lo, hi = mean_ci(values)
            lx = lonx["word_auprc"].get((method, model), (np.nan,) * 3 + (0,))
            rows.append({**base, "metric": "word_auprc",
                         "lonxplain": lx[0], "lonxplain_lo": lx[1],
                         "lonxplain_hi": lx[2], "n_lonxplain": lx[3],
                         "ood": value, "ood_lo": lo, "ood_hi": hi,
                         "n_ood": len(values), "n_ood_pos": len(values),
                         "n_ood_neg_annotated": 0, "n_ood_neg_screen": 0,
                         # a random ranker's expected precision on these
                         # phases: gold words over scored words
                         "chance_ood": (float(np.mean(np.asarray(n_gold)
                                                      / np.asarray(n_scored)))
                                        if values else float("nan"))})

            # -- faithfulness: the word-deletion margin on the positives
            if deletion is not None and len(positives):
                sub = deletion[deletion["id"].isin(set(positives["phase_id"]))]
                margin = (_margins(sub, ("method",), positive_gap_only=True)
                          if len(sub) else pd.DataFrame())
            else:
                margin = pd.DataFrame()
            if len(margin):
                m = margin.iloc[0]
                value, lo, hi, n = (float(m["margin"]), float(m["margin_lo"]),
                                    float(m["margin_hi"]), int(m["n_posts"]))
            else:
                value, lo, hi, n = float("nan"), float("nan"), float("nan"), 0
            lx = lonx["aopc"].get((method, model), (np.nan,) * 3 + (0,))
            rows.append({**base, "metric": "aopc",
                         "lonxplain": lx[0], "lonxplain_lo": lx[1],
                         "lonxplain_hi": lx[2], "n_lonxplain": lx[3],
                         "ood": value, "ood_lo": lo, "ood_hi": hi,
                         "n_ood": n, "n_ood_pos": n,
                         "n_ood_neg_annotated": 0, "n_ood_neg_screen": 0,
                         "chance_ood": 0.0})
    return pd.DataFrame(rows)


def upsert(table: pd.DataFrame) -> None:
    """Merge this run's (method, model) rows into results/ood_transfer.csv."""
    if table.empty:
        print("nothing to write")
        return
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        keys = set(zip(table["method"], table["model"]))
        old = old[~old.apply(lambda r: (r["method"], r["model"]) in keys,
                             axis=1)]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["model", "metric", "annotator", "method"],
                              kind="stable")
    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(table)} rows)")


# ================================================================== OOD.md

OOD_MD = STAGE3 / "OOD.md"
AGREEMENT_CSV = STAGE3 / "annotator_agreement.csv"
METRIC_TITLE = {
    "auroc": ("Detection AUROC", "all 844 screened phases: the annotator's "
              "label on the 34 annotated ones, 0 on the 810 the screen did "
              "not flag; chance 0.5"),
    "auroc_annotated": ("Detection AUROC, annotated phases only",
                        "the 34 phases both annotators read; annotator 1 "
                        "leaves 2 negatives and annotator 2 leaves 12, so "
                        "the intervals are wide"),
    "word_auprc": ("Word AUPRC vs the annotator's spans",
                   "the method's own word attribution against the located "
                   "quotes, on the phases that annotator ticked; chance is "
                   "the gold-word base rate given per column"),
    "aopc": ("Word-deletion \u0394AOPC", "target minus the count-matched "
             "random-word control, same k grid as Stage 2, on the phases "
             "that annotator ticked; 0 = no better than random deletion"),
}
VIEW_TITLE = {"annotator1": "annotator 1", "annotator2": "annotator 2",
              "pooled": "pooled"}


def _cell(value, lo, hi, digits=3) -> str:
    if not np.isfinite(value):
        return "\u2014"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{value:.{digits}f} [{lo:.{digits - 1}f}, {hi:.{digits - 1}f}]"
    return f"{value:.{digits}f}"


def write_ood_md(table: pd.DataFrame, annotations: pd.DataFrame) -> None:
    """results/stage3/OOD.md: agreement, then one table per metric."""
    from plot.style import DISPLAY  # the study's short method names

    agreement = pd.read_csv(AGREEMENT_CSV).set_index("statistic")
    n = len(annotations)
    a1 = int(annotations["annotator1_label"].sum())
    a2 = int(annotations["annotator2_label"].sum())
    agreed = int(annotations["agreed"].sum())
    pa = agreement.loc["percent_agreement"]
    jac = agreement.loc["span_jaccard"]
    f1 = agreement.loc["span_f1"]

    lines = [
        "# Stage 3: the Stage 1 detectors on older-adult conversations",
        "",
        "Rendered by `src.stage3.ood_detection --collect` from "
        "`results/ood_transfer.csv` and `results/stage3/annotator_agreement.csv`; "
        "do not edit by hand. Every method is its fold-0 cell carried over with "
        "no retraining and no re-cut threshold, reading the older adult's turns "
        "in a conversation phase. Intervals are 2,000-draw bootstraps over "
        "phases (over posts for the LonXplain column).",
        "",
        "## Annotator agreement",
        "",
        f"Two expert annotators read the {n} phases the screen flagged. "
        f"Annotator 1 ticked {a1}, annotator 2 ticked {a2}; they agree on "
        f"{agreed} of {n} phases.",
        "",
        "| statistic | value | 95% CI | n |",
        "|---|---|---|---|",
        f"| percent agreement on the label | {pa['value']:.3f} | "
        f"[{pa['lo']:.2f}, {pa['hi']:.2f}] | {int(pa['n_phases'])} phases |",
        f"| span overlap, word-level Jaccard | {jac['value']:.3f} | \u2014 | "
        f"{int(jac['n_phases'])} phases both ticked and quoted |",
        f"| span overlap, word-level F1 | {f1['value']:.3f} | \u2014 | "
        f"{int(f1['n_phases'])} phases both ticked and quoted |",
        "",
        "Both labels and both span sets are kept as columns in "
        "`results/stage3/annotations.csv`; nothing is adjudicated. Each metric "
        "below is read against annotator 1, against annotator 2, and on a "
        "pooled set in which a disputed phase appears once under each reading "
        "and an agreed phase once.",
        "",
        "## Results",
        "",
        "Columns: the LonXplain value from Stages 1/2, then the conversations "
        "under each view. `n` is the number of phases the OOD value is read "
        "over, per view, in the order annotator 1 / annotator 2 / pooled.",
    ]
    methods = sorted(set(table["method"]),
                     key=lambda m: list(DISPLAY).index(m) if m in DISPLAY else 99)
    for metric, (title, note) in METRIC_TITLE.items():
        sub = table[table["metric"] == metric]
        if sub.empty:
            continue
        lines += ["", f"### {title}", "", note[0].upper() + note[1:] + ".", ""]
        header = ["Method", "Backbone", "LonXplain"] + \
                 [f"Conversations, {VIEW_TITLE[v]}" for v in VIEWS] + ["n"]
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        for model in MODELS:
            for method in methods:
                rows = sub[(sub["model"] == model) & (sub["method"] == method)]
                if rows.empty:
                    continue
                by_view = {r["annotator"]: r for _, r in rows.iterrows()}
                first = rows.iloc[0]
                cells = [DISPLAY.get(method, method).replace("\n", ""),
                         model,
                         _cell(first["lonxplain"], first["lonxplain_lo"],
                               first["lonxplain_hi"])]
                for view in VIEWS:
                    r = by_view.get(view)
                    cells.append(_cell(r["ood"], r["ood_lo"], r["ood_hi"])
                                 if r is not None else "\u2014")
                cells.append(" / ".join(str(int(by_view[v]["n_ood"]))
                                        if v in by_view else "\u2014"
                                        for v in VIEWS))
                lines.append("| " + " | ".join(cells) + " |")
        if metric == "word_auprc":
            chance = {v: float(sub[sub["annotator"] == v]["chance_ood"].mean())
                      for v in VIEWS}
            lines += ["", "Chance (random ranker): LonXplain 0.083; "
                      + "; ".join(f"{VIEW_TITLE[v]} {chance[v]:.3f}"
                                  for v in VIEWS) + "."]
    OOD_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OOD_MD.relative_to(ROOT)}")


# ======================================================================= CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_method_args(ap, choices=METHODS)
    ap.add_argument("--corpus", action="append", default=None,
                    help="repeatable; defaults to every conversation corpus")
    ap.add_argument("--collect", action="store_true",
                    help="build results/ood_transfer.csv from the rows already "
                         "on disk, for every model unless --model is given; "
                         "scores nothing and needs no GPU")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N screened phases plus every "
                         "annotated one (a smoke run; written under a smoke/ "
                         "directory so it can never be mistaken for the study)")
    ap.add_argument("--chunk", type=int, default=32,
                    help="phases held as token vectors at once (default 32)")
    ap.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                    help="forward-pass batch size (default 8)")
    ap.add_argument("--allow-model-process", action="store_true",
                    dest="allow_model_process")
    add_device_args(ap)
    args = ap.parse_args()

    methods = (selected_methods(args, choices=METHODS) if args.method
               else list(STUDY_METHODS))
    from src.stage3.prepare_conversations import CORPORA

    corpora = args.corpus or list(CORPORA)
    unknown = set(corpora) - set(CORPORA)
    if unknown:
        ap.error(f"unknown corpus {sorted(unknown)}; choose from {list(CORPORA)}")

    print(f"loading the screened phases from {corpora}")
    phases, turns_of = load_phases(corpora)
    annotations = load_annotations(required=True)
    annotated = set(annotations["phase_id"])
    missing = annotated - set(phases["phase_id"])
    if missing:
        raise SystemExit(f"{len(missing)} annotated phases are not in the "
                         f"screened corpus: {sorted(missing)[:5]}")
    print(f"  {len(phases)} screened phases, {len(annotated)} annotated by "
          "both clinicians")
    smoke = args.limit is not None

    if args.collect:
        models = [resolve_model(args.model)] if args.model else list(MODELS)
        parts = []
        for model in models:
            print(f"collecting {model}")
            parts.append(collect_metrics(model, methods, phases, annotations,
                                         smoke=smoke))
        table = pd.concat(parts, ignore_index=True)
        if table.empty:
            print("nothing to collect")
            return 1
        if smoke:
            print(table.to_string())
            return 0
        upsert(table)
        write_ood_md(pd.read_csv(OUT_CSV), annotations)
        shown = table[table["metric"].isin(("auroc", "word_auprc", "aopc"))]
        print(shown.pivot_table(index=["model", "method"],
                                columns=["metric", "annotator"],
                                values="ood").round(3).to_string())
        record_decision("stage3.ood_transfer", {
            "table": str(OUT_CSV.relative_to(ROOT)),
            "views": list(VIEWS),
            "pooled_rule": "a disputed phase appears once per reading; an "
                           "agreed phase once (duplicate disputed instances)",
            "auroc_negatives": "annotated 0s plus every screen-negative "
                               "phase as 0; auroc_annotated uses the "
                               "annotated phases alone",
            "word_auprc": "own word attribution vs the clinician's located "
                          "quotes, phases that clinician ticked",
            "aopc": "word-deletion margin over the count-matched random "
                    "control, phases that clinician ticked, "
                    "src.stage2.word_deletion._margins",
            "deploy_fold": DEPLOY_FOLD,
            "methods": methods,
            "models": models,
        })
        return 0

    model = resolve_model(args.model)
    cfg = load_config()
    pin_visible_devices(args.devices, need_gpu=True,
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)
    from src.utils.common import require_model_process_approval

    require_model_process_approval(cfg, cli_override=args.allow_model_process,
                                   command="src.stage3.ood_detection")
    if smoke:
        head = phases.head(args.limit)
        phases = pd.concat([head, phases[phases["phase_id"].isin(annotated)
                                         & ~phases["phase_id"].isin(
                                             head["phase_id"])]],
                           ignore_index=True)
        print(f"  SMOKE RUN: {len(phases)} phases", flush=True)

    results = {}
    dense = [m for m in methods if m in DENSE]
    if dense:
        results |= deploy_dense_probes(dense, model, phases, turns_of,
                                       annotated, cfg, chunk=args.chunk,
                                       batch_size=args.batch_size)
    if "sae_probe" in methods:
        results |= deploy_sae_probe(model, phases, turns_of, annotated, cfg,
                                    allow=args.allow_model_process,
                                    chunk=args.chunk,
                                    batch_size=args.batch_size)
    arms = [m for m in methods if m in ARMS]
    if arms:
        print("fine-tuning arms:", flush=True)
        results |= deploy_arms(arms, model, phases, turns_of, annotated, cfg,
                               batch_size=args.batch_size)
    text = [m for m in methods if m in TEXT]
    if text:
        print("reference detectors:", flush=True)
        results |= deploy_text_models(text, model, phases, turns_of, annotated,
                                      cfg, batch_size=args.batch_size)
    if not results:
        print("no method produced a score")
        return 1

    for method, result in results.items():
        frame = phase_rows(method, model, phases, result, annotations)
        tables = ((phase_path(method, model, smoke), frame),
                  (word_path(method, model, smoke),
                   pd.DataFrame(result["word_rows"], columns=WORD_ROW_COLUMNS)),
                  (delete_path(method, model, smoke),
                   pd.DataFrame(result["delete_rows"])))
        for path, table in tables:
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            print(f"wrote {path.relative_to(ROOT)} ({len(table)} rows)")
        n_flag = int(frame["flagged"].sum())
        print(f"  {method}: flagged {n_flag} of {len(frame)} phases "
              f"({int((frame['flagged'] & frame['in_clinician_script']).sum())}"
              f" of the {int(frame['in_clinician_script'].sum())} in the "
              "clinician script)")

    if not smoke:
        record_decision(f"stage3.ood_detection.{model}", {
            "unit": UNIT,
            "scored_speaker": SPEAKER,
            "deploy_fold": DEPLOY_FOLD,
            "thresholds": "transferred unchanged from the Stage 1 training "
                          "folds; decide `flagged` only",
            "word_score_aggregation": cfg["audit"]["word_score_aggregation"],
            "faithfulness": {k: cfg["faithfulness"][k]
                             for k in ("k_grid", "random_draws", "controls")},
            "corpora": corpora,
            "methods": sorted(results),
            "n_phases": int(len(phases)),
            "n_annotated": len(annotated),
            "flagged_per_method": {
                m: int((r["scores"] >= r["threshold"]).sum())
                for m, r in results.items()},
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
