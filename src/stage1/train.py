"""Train every post-level loneliness detector on LonXplain.

How to run. --method defaults to every method and --model defaults to gemma2,
so the commands below are the minimal ones; add `--model llama` for the family
control. One command per method:

    # frozen probes — CPU, fit on cached activations, no model loaded, no gate
    uv run python -m src.stage1.train --method linear_probe
    uv run python -m src.stage1.train --method mlp_probe
    uv run python -m src.stage1.train --method attention_probe
    uv run python -m src.stage1.train --method sae_probe

    # fine-tuning arms + the conventional-recipe anchor — GPU, gated
    uv run python -m src.stage1.train --method full_ft --allow-model-process
    uv run python -m src.stage1.train --method lp_ft   --allow-model-process
    uv run python -m src.stage1.train --method attn_ft    --allow-model-process
    uv run python -m src.stage1.train --method attn_lp_ft --allow-model-process
    uv run python -m src.stage1.train --method anchor  --allow-model-process

    # reference baselines
    uv run python -m src.stage1.train --method glove_bigru        # GPU, no gate,
                                                                  # trains all ten folds
    uv run python -m src.stage1.train --method mentalbert --allow-model-process
    uv run python -m src.stage1.train --method prompting  --allow-model-process
    uv run python -m src.stage1.train --method prompting --model opus --allow-model-process
                                                                  # Claude Opus 5
                                                                  # over the Batch
                                                                  # API; prompting
                                                                  # is the only
                                                                  # method it runs

    # groupings
    uv run python -m src.stage1.train --method linear_probe mlp_probe attention_probe sae_probe
    uv run python -m src.stage1.train --method full_ft lp_ft attn_ft attn_lp_ft --allow-model-process
    uv run python -m src.stage1.train --allow-model-process        # everything, gemma2
    uv run python -m src.stage1.train --model llama --allow-model-process

glove_bigru and mentalbert ignore --model entirely: their trainers take no
backbone argument and their rows are labelled glove-300d / mentalbert, so
`--method glove_bigru` on its own is the whole command.

Then aggregate:  uv run python -m src.stage1.detection

Methods (registry in src/utils/common.py):
    probes     linear_probe, mlp_probe, attention_probe, sae_probe,
               afr_probe
    baselines  full_ft, lp_ft, attn_ft, attn_lp_ft,
               midlayer_lp_ft, midlayer_attn_lp_ft,
               lora_ft, lora_lp_ft, attn_lora_ft, attn_lora_lp_ft,
               span_ft, attn_span_ft, anchor,
               mentalbert, glove_bigru, prompting

The LoRA arms are the same 2x2 with low-rank adapters instead of full
weight updates (merged into the checkpoint, so downstream reads them as
any other arm); the span arms are full_ft / attn_ft with the expert span
added to the loss as a token-level target (fine_tune.ARMS).

The four fine-tuning arms are the pooling x head-initialization 2x2
(src/baselines/fine_tune.ARMS): full_ft and lp_ft mean-pool, attn_ft and
attn_lp_ft pool with a learned query (fine_tune.AttentionPooledClassifier).
attn_lp_ft is AP-LP-FT: the frozen attention probe is solved on the final-layer
states first, installed as the query and head, and only then fine-tuned.

What training writes, named and placed by the one rule in src/utils/common.py:
{stage}_{method}_{model}[_fold{k}].{ext}, in the directory that stage prefix
owns. A cell's scores and its recipe sidecar therefore share a stem across two
directories -- results/detection/detect_ft_gemma_fold0.parquet holds the scores,
results/train/train_ft_gemma_fold0.json the recipe that produced them -- which
is how src.stage1.detection pairs them.

  results/detection/
    probes           detect_{method}_{model}.parquet — one out-of-fold score
                     per study-fold post (folds 0 2 4 6 8), with the `pred` it
                     implies and the `threshold` behind it, fitted on that
                     fold's training posts
    FT arms/anchor   detect_{method}_{model}_fold{k}.parquet
    mentalbert       detect_mentalbert_{encoder}.{parquet,json}
    glove_bigru      detect_glove_{model}.parquet, assembled once all ten
                     detect_glove_{model}_fold{k}.parquet pieces exist
    prompting        detect_prompt_{model}.csv

  results/train/
    FT arms/anchor   train_{method}_{model}_fold{k}.json, the recipe, plus
                     checkpoints/ (fold 0 always; --keep-checkpoint for the
                     rest). Training computes NO metric: every Stage 2 metric
                     loads the retained checkpoint from its own CLI, so what a
                     metric measures is decided by that metric rather than by
                     whichever training run happened to go first.
    glove_bigru      train_glove_{model}_fold{k}.json, one per fold

`src.stage1.detection` turns these into results/detection.csv.

GPU methods that load a study backbone (FT arms, mentalbert, prompting)
require --allow-model-process. glove_bigru also takes a card, but it trains a
small net from scratch and loads no study model, so it is ungated.
Probes fit on cached activations and are pinned to CPU unless --gpu is given,
so the published fits are reproduced exactly.

Every GPU method is a single-card recipe. --devices picks that card: the
default `auto` takes the emptiest GPU with enough free memory for this
backbone, so a run does not land on a card somebody else is filling. Pass
`--devices 1,2` (or set CUDA_VISIBLE_DEVICES yourself) to choose explicitly.
"""

import argparse
import sys

from src.utils.common import (
    artifact_path,
    ALL,
    ARM_OF,
    DEFAULT_MODEL,
    is_api_model,
    MENTALBERT_ENCODER,
    DENSE_PROBE_LAYER,
    DENSE_PROBES,
    N_FOLDS,
    RESULTS,
    ROOT,
    SAE_PROBE_LAYER,
    add_device_args,
    add_method_args,
    best_f1_threshold,
    checkpoint_path,
    detection_metrics,
    load_config,
    pin_visible_devices,
    record_decision,
    require_model_process_approval,
    resolve_model,
    score_to_pred,
    selected_methods,
    span_eval_ids,
    subsample,
    variants_of,
)


DEFAULT_FOLDS = [0, 2, 4, 6, 8]
GPU_METHODS = set(ARM_OF) | {"mentalbert", "prompting", "glove_bigru"}

# What one card has to have free for a run to get off the ground: bf16 weights
# + bf16 grads + 8-bit Adam moments, plus activations at batch 16 / 512 tokens
# under checkpointing. Measured peaks are ~49 GB (llama31-8b) and ~16 GB
# (gemma-2-2b); the headroom below keeps --devices auto from picking a card
# that is only nominally free.
GPU_MIN_FREE_GB = {"llama31-8b": 60.0, "gemma-2-2b": 24.0}

# glove_bigru loads no backbone at all — a BiGRU over 300-d vectors at batch 16
# / 512 tokens peaks near 1 GB — so a glove-only invocation must not be held to
# a backbone's headroom, or it would refuse cards it fits on many times over.
GLOVE_MIN_FREE_GB = 4.0


def scores_path(method: str, model: str):

    return artifact_path("detect", method, model)


# ------------------------------------------------------------- frozen probes


def _upsert_scores(out, table, variants):
    """Write one probe's score file, replacing only the variants just fitted.

    No method currently publishes more than one variant, so this is a plain
    overwrite in practice. It stays variant-aware because a run that fitted
    one variant of a split method must not delete the others' rows, or a later
    `src.stage1.detection` would silently report a table with a missing arm
    rather than failing.
    """
    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_parquet(out)
        if "variant" not in old.columns:
            old["variant"] = ""
        old = old[~old["variant"].isin(list(variants))]
        if len(old):
            table = pd.concat([old, table], ignore_index=True)
    table.to_parquet(out, index=False)


def training_threshold(scores, labels) -> float:
    """The cut that maximises present-class F1 on the training folds.

    Fitted on the training posts only -- the held fold's labels are read by
    the metric and nowhere else -- and measured on the same quantity the
    held-out scores are, so the cut transfers with no rescaling step. This is
    the recipe `_score_detection` uses for the fine-tuning arms, which is what
    makes a probe's f1/fpr/fnr and an arm's the same measurement rather than
    two things that share a column name.
    """
    import numpy as np

    return float(best_f1_threshold(np.asarray(scores, dtype=np.float64),
                                   np.asarray(labels).astype(int)))


def train_dense_probes(methods: list[str], model: str, folds, seed: int,
                       layer: int | None):
    """Fit each probe that reads cached activations, out of fold, and cache
    its detection scores.

    A method that splits its published rows (`VARIANTS_OF` in
    src/utils/common.py; none do today) is fitted once per variant against the
    same activations and the same folds, and the variant travels in the score
    file's `variant` column. Everything else writes a single block with an
    empty variant.
    """
    import numpy as np
    import pandas as pd

    from src import probe as probing

    cfg = load_config()
    from src.utils.common import probe_layer_of

    default_layer = layer
    ctxs: dict = {}

    def ctx_at(at: int):
        if at not in ctxs:
            print(f"loading {model} layer {at} activations", flush=True)
            ctxs[at] = probing.build_context(model, at, cfg, seed)
        return ctxs[at]

    for method in methods:
        # A final-layer probe reads a different cache file than the rest
        # (common.FINAL_PROBE_LAYER); every other probe keeps the pinned layer
        # or the --layer override.
        layer = probe_layer_of(method, model, default_layer)
        ctx = ctx_at(layer)
        wanted = list(variants_of(method))
        rows = []
        for variant in wanted:
            tag = f"{method}[{variant}]" if variant else method
            provenance = {}
            for fold in folds:
                fold_ctx = probing.for_fold(ctx, fold)
                fitted = probing.FITTERS[method](fold_ctx)
                held = [i for i, f in enumerate(ctx.fold_col) if f == fold]
                scores = np.asarray(fitted.post_scores(np.array(held)),
                                    dtype=np.float64)
                assert np.isfinite(scores).all(), f"{tag} fold {fold}: non-finite score"
                train_idx = np.asarray(fold_ctx.train_idx, dtype=int)
                thr = training_threshold(fitted.post_scores(train_idx),
                                         ctx.y[train_idx])
                rows.append(pd.DataFrame({
                    "method": method, "group": "probe", "model": model,
                    "variant": variant, "layer": layer, "fold": fold,
                    "id": [ctx.all_ids[i] for i in held],
                    "label": ctx.posts["label"].to_numpy()[held],
                    "score": scores,
                    "threshold": thr,
                    "pred": score_to_pred(scores, thr),
                    **provenance,
                    **(fitted.provenance or {}),
                }))
                print(f"  {tag} fold {fold}: fitted on {len(fold_ctx.train_idx)}"
                      f", scored {len(held)}", flush=True)
        out = scores_path(method, model)
        _upsert_scores(out, pd.concat(rows, ignore_index=True), wanted)
        print(f"wrote {out.name}")


def train_sae_attention_probe(model: str, folds, seed: int,
                              layer: int | None, allow: bool):
    """Fit a learned query over PER-TOKEN dictionary latents, out of fold.

    Differs from train_sae_probe in what the encode pass has to keep. That one
    pools inside the pass and stores one row per post; attention pooling is
    fitted, so the tokens have to survive the pass -- `token_post_ids` is the
    whole corpus here rather than the plausibility subset, which is ~58M
    non-zeros (~460 MB) at the measured ~99 live latents per token. The pass
    runs once and is cached under the `all` tag.
    """
    import numpy as np
    import pandas as pd

    from src.probe import sae_attention_probe, sae_probe
    from src.utils.data_prep import load_corpus

    cfg = load_config()
    pinned_cache = SAE_PROBE_LAYER[model]
    layer = layer if layer is not None else pinned_cache
    sae_layer = int(cfg["sae"][model]["layer"])
    print(f"sae_attention_probe on {model}: dictionary layer {sae_layer} "
          f"(blocks.{sae_layer}.hook_resid_post), read from cached layer "
          f"{layer}", flush=True)
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    fold_col = posts["fold"].to_numpy()

    tok_path = sae_probe.token_latent_cache_path(model, "lonxplain", layer,
                                                 "all")
    sae = None
    if not tok_path.exists():
        sae, _, _ = sae_probe.load_pinned_sae(
            cfg, model, device=None, allow=allow, command="src.stage1.train")
    _, tokens = sae_probe.encode_posts_to_latents(
        sae, model, "lonxplain", layer, post_ids=all_ids, pool="mean",
        token_post_ids=all_ids, token_tag="all")

    rows = []
    for fold in folds:
        train_ids = [all_ids[i] for i in np.flatnonzero(fold_col != fold)]
        held_ids = [all_ids[i] for i in np.flatnonzero(fold_col == fold)]
        probe = sae_attention_probe.fit(tokens, posts, train_ids, seed, cfg)
        # Enforce the identity once per fold rather than trusting it, exactly
        # as attention_probe does.
        sae_attention_probe.verify_decomposition(
            probe, tokens.rows[train_ids[0]])
        scores = sae_attention_probe.score_posts(probe, tokens, held_ids)
        thr = training_threshold(
            sae_attention_probe.score_posts(probe, tokens, train_ids),
            (posts.set_index("id").loc[train_ids, "label"] == "present")
            .to_numpy().astype(int))
        held = np.flatnonzero(fold_col == fold)
        rows.append(pd.DataFrame({
            "method": "sae_attention_probe", "group": "probe", "model": model,
            "variant": "", "layer": layer, "sae_layer": sae_layer,
            "fold": fold,
            "id": held_ids,
            "label": posts["label"].to_numpy()[held],
            "score": scores,
            "threshold": thr,
            "pred": score_to_pred(scores, thr),
        }))
        print(f"  sae_attention_probe fold {fold}: "
              f"{probe['n_nonzero_coef']} non-zero coefficients of "
              f"{probe['d_sae']}", flush=True)
    out = scores_path("sae_attention_probe", model)
    _upsert_scores(out, pd.concat(rows, ignore_index=True), [""])
    print(f"wrote {out.name}")


def train_sae_probe(model: str, folds, seed: int, layer: int | None,
                    allow: bool):
    """Fit the L1 SAE probe on mean-pooled frozen-dictionary latents, out of fold.

    `C` is fixed at 1.0 -- the study's pinned penalty, chosen a priori rather
    than on labels, as everywhere else in src/probe/PROBES.md.
    """
    pool = "mean"
    import numpy as np
    import pandas as pd

    from src.probe import sae_probe
    from src.utils.data_prep import load_corpus

    cfg = load_config()
    pinned_cache = SAE_PROBE_LAYER[model]
    layer = layer if layer is not None else pinned_cache
    sae_layer = int(cfg["sae"][model]["layer"])
    if layer == pinned_cache:
        print(f"sae_probe on {model}: dictionary layer {sae_layer} "
              f"(blocks.{sae_layer}.hook_resid_post), read from cached layer "
              f"{layer}", flush=True)
    else:
        print(f"sae_probe on {model}: --layer override, reading cached layer "
              f"{layer}; the pinned dictionary is layer {sae_layer} (cached "
              f"{pinned_cache})", flush=True)
    posts, _ = load_corpus("lonxplain")
    all_ids = posts["id"].tolist()
    y = (posts["label"] == "present").to_numpy().astype(int)
    fold_col = posts["fold"].to_numpy()

    sae = None
    if not sae_probe.latent_cache_path(model, "lonxplain", layer, pool).exists():
        sae, _, _ = sae_probe.load_pinned_sae(
            cfg, model, device=None, allow=allow, command="src.stage1.train")
    latents, _ = sae_probe.encode_posts_to_latents(
        sae, model, "lonxplain", layer, post_ids=all_ids, pool=pool)
    X = latents.X

    rows = []
    method = "sae_probe"
    for fold in folds:
        train_idx = np.flatnonzero(fold_col != fold)
        held = np.flatnonzero(fold_col == fold)
        C = 1.0
        probe = sae_probe.fit_sae_probe(X[train_idx], y[train_idx],
                                        seed=seed, C=C, pool=pool)
        scores = np.asarray(sae_probe.score_sae_probe(probe, X[held]),
                            dtype=np.float64)
        thr = training_threshold(sae_probe.score_sae_probe(probe, X[train_idx]),
                                 y[train_idx])
        rows.append(pd.DataFrame({
            "method": method, "group": "probe", "model": model,
            "variant": "", "layer": layer, "sae_layer": sae_layer,
            "fold": fold,
            "id": [all_ids[i] for i in held],
            "label": posts["label"].to_numpy()[held],
            "score": scores,
            "threshold": thr,
            "pred": score_to_pred(scores, thr),
        }))
        print(f"  {method} fold {fold}: C={C:g}, {probe.n_nonzero_coef} "
              f"non-zero coefficients of {probe.d_sae}", flush=True)
    out = scores_path(method, model)
    # One block, empty variant: the probe publishes a single row per
    # (method, model), so a re-fit replaces its scores rather than adding a row.
    _upsert_scores(out, pd.concat(rows, ignore_index=True), [""])
    print(f"wrote {out.name}")


# ---------------------------------------------------------- fine-tuning arms


DEGENERATE_AUROC = 0.55
# Training produces WEIGHTS and the held-fold scores that prove they trained.
# It computes no metric: every Stage 2 metric now loads the retained checkpoint
# itself (src.stage2.*), so what a metric measures is decided by the metric's
# own CLI rather than by whichever training run happened to run first.
ARM_STAGES = ["detection"]


def _cell_paths(arm: str, model: str, cell: int, *, smoke: bool = False):
    """Where one trained cell's evidence lands: scores and recipe sidecar.

    They share a stem but sit in different directories, one per stage prefix
    (src/utils/common.py) -- detection scores under results/detection/, the
    recipe that produced them under results/train/.

    Training writes weights and held-fold scores only. Stage 2 metrics read the
    retained checkpoint themselves, so nothing here is a metric.
    """
    return {
        "detection": artifact_path("detect", arm, model, fold=cell, smoke=smoke),
        "train_info": artifact_path("train", arm, model, fold=cell, ext="json",
                                    smoke=smoke),
    }


def _archive_existing_outputs(paths: dict, active_stages=()) -> None:
    """Preserve existing artifacts before an explicitly forced retrain.

    Artifacts the rerun will regenerate are copied (the fresh run overwrites
    them anyway). Artifacts it will NOT regenerate are MOVED, because leaving
    them beside a new train sidecar would let src.stage1.detection publish
    stale rows as though the retrained weights had produced them.
    """
    import shutil

    # One archive for all of it: a forced retrain supersedes a cell's scores
    # and its recipe together, and they no longer share a directory.
    archive = RESULTS / "legacy_pre_v3"
    active = set(active_stages)
    existing = [(stage, path) for stage, path in paths.items() if path.exists()]
    if not existing:
        return
    archive.mkdir(parents=True, exist_ok=True)
    for stage, path in existing:
        # Archive names are unique per generation: a second --force-retrain of
        # the same cell must not overwrite what the first one preserved.
        target = archive / path.name
        generation = 1
        while target.exists():
            target = archive / f"{path.stem}.gen{generation}{path.suffix}"
            generation += 1
        if stage in active or stage == "train_info":
            shutil.copy2(path, target)
        else:
            shutil.move(str(path), str(target))


def _train_cell(arm: str, spec: dict, model_name: str, cell: int, cfg: dict,
                seed: int, posts):
    """Train one (arm, fold) at full budget; returns the model + held fold."""
    from src.baselines import fine_tune as ft
    from src.utils.common import best_f1_threshold

    fcfg = dict(cfg["full_finetune"])
    lr = fcfg["anchor_lr"] if spec.get("use_anchor_lr") else None
    pooling = spec.get("pooling", "mean")
    # The head's site. The arm spec cannot carry the index because it is
    # per-backbone, so it names the pin and this resolves it -- the same
    # DENSE_PROBE_LAYER the frozen probes read, so a mid-layer arm and its
    # frozen partner are on one site by construction rather than by a number
    # copied into two places.
    head_at = spec.get("head_at")
    if head_at is None:
        head_layer = None
    elif head_at == "dense_probe_layer":
        head_layer = DENSE_PROBE_LAYER[model_name]
    else:
        raise ValueError(f"unknown head_at {head_at!r} in arm spec for {arm}")
    if pooling == "attention":
        # The attention arms' LP stage runs the FROZEN attention probe's solver
        # (src/probe/attention_probe.solve), so its settings travel with the
        # recipe and land in this cell's train_*.json sidecar rather than being
        # read from a global at some unrecorded moment.
        fcfg["attention_pool"] = dict(cfg["attention_pool"])
    objective = spec.get("objective", "bce")
    # The same rule for the two other recipe blocks: what an arm trains under
    # is copied into its recipe so the sidecar records it.
    if spec["recipe"] == "lora":
        fcfg["lora"] = dict(cfg["lora"])
    if objective == "span":
        fcfg["span_supervision"] = dict(cfg["span_supervision"])
    text_by_id = posts.set_index("id")["text"]
    label_by_id = posts.set_index("id")["label"]
    train_ids = subsample(posts, cell, seed, ALL)
    train_texts = [text_by_id.loc[p] for p in train_ids]
    held = posts[posts["fold"] == cell]
    y = (label_by_id.loc[train_ids] == "present").astype(int).to_numpy()
    spans = None
    if objective == "span":
        # The expert span of each TRAINING post, in post character offsets;
        # -1 where there is none (every absent post, and the present posts
        # whose span did not align). The held fold's spans are never read.
        by_id = posts.set_index("id")
        spans = [(int(by_id.loc[p, "span_start"]), int(by_id.loc[p, "span_end"]))
                 for p in train_ids]

    site = "final layer" if head_layer is None else f"layer {head_layer}"
    print(f"  training {arm} fold{cell} on {len(train_ids)} posts "
          f"({pooling} pooling, {spec['init']} head at {site}, "
          f"{spec['recipe']} recipe, {objective} objective)", flush=True)
    model, tok, info = ft.train(
        model_name, train_texts, y, fcfg, seed=seed,
        init=spec["init"], recipe=spec["recipe"], pooling=pooling, lr=lr,
        log_every=100, head_layer=head_layer, objective=objective,
        spans=spans,
    )
    train_scores = ft.predict(model, tok, train_texts, fcfg, batch_size=16)
    # The threshold is fitted on the training folds only; the held fold's
    # labels are read by the metric and nowhere else.
    info["threshold"] = float(best_f1_threshold(train_scores, y))
    info["n_train"] = len(train_ids)
    return model, tok, fcfg, info, held


def _score_detection(model, tok, fcfg, arm, model_name, cell, info, held):
    """Held-out-fold scores at the threshold fitted on the training folds."""
    import pandas as pd

    from src.baselines import fine_tune as ft
    from src.utils.data_prep import load_corpus

    thr = info["threshold"]
    s = ft.predict(model, tok, held["text"].tolist(), fcfg, batch_size=16)
    return pd.DataFrame({
        "arm": arm, "model": model_name, "corpus": "lonxplain",
        "role": "in_domain", "fold": cell,
        "id": held["id"].to_numpy(), "label": held["label"].to_numpy(),
        "post_score": s, "post_pred": score_to_pred(s, thr),
    })


def _run_cell(arm, spec, model_name, cell, cfg, seed, posts, stages,
              limit_eval=None, force_retrain=False, smoke=False,
              keep_checkpoint=False):
    import json
    import shutil
    import time

    import torch

    from src.baselines import fine_tune as ft
    from src.utils.common import detection_metrics

    paths = _cell_paths(arm, model_name, cell, smoke=smoke)
    active_stages = (
        list(dict.fromkeys(stages))
        if force_retrain
        else [s for s in stages if not paths[s].exists()]
    )
    if not active_stages:
        print(f"  {arm} fold{cell}: already done, skipping", flush=True)
        return
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checkpoint = checkpoint_path(arm, model_name, cell, smoke=smoke)
    train_info_path = paths["train_info"]
    loaded_checkpoint = (
        not force_retrain and checkpoint.exists() and train_info_path.exists()
    )
    if loaded_checkpoint:
        print(f"  loading existing checkpoint {checkpoint}", flush=True)
        model, tok = ft.load_checkpoint(checkpoint, model_name)
        fcfg = dict(cfg["full_finetune"])
        with open(train_info_path) as handle:
            info = json.load(handle)
        held = posts[posts["fold"] == cell]
    else:
        if force_retrain and not smoke:
            _archive_existing_outputs(paths, active_stages)
        # Fold-0 weights are the only checkpoint kept long-term, and a retrain
        # takes 45-70 min. Move the old checkpoint aside rather than deleting
        # it, so a crashed or OOM-killed retrain leaves a recoverable copy.
        stashed = None
        if force_retrain and checkpoint.exists():
            stashed = checkpoint.parent / f"legacy_pre_v3_{checkpoint.name}"
            if stashed.exists():
                shutil.rmtree(stashed)
            checkpoint.rename(stashed)
        model, tok, fcfg, info, held = _train_cell(
            arm, spec, model_name, cell, cfg, seed, posts)
        if cell == 0 or keep_checkpoint:
            # Fold 0 is retained by default because every Stage 2 metric loads
            # it. --keep-checkpoint retains the rest: a metric can only score a
            # cell whose weights still exist, and retaining is far cheaper than
            # retraining to score.
            ft.save_checkpoint(model, tok, checkpoint)
        if stashed is not None and checkpoint.exists():
            # New weights are on disk; the stash has done its job.
            shutil.rmtree(stashed)
        with open(train_info_path, "w") as handle:
            json.dump(info, handle, indent=2)

    if "detection" in active_stages:
        det = _score_detection(model, tok, fcfg, arm, model_name, cell, info,
                               held)
        det.to_parquet(paths["detection"], index=False)
        y_held = (det["label"] == "present").to_numpy().astype(int)
        pred_held = (det["post_pred"] == "present").to_numpy().astype(int)
        mets = detection_metrics(y_held, pred_held,
                                 det["post_score"].to_numpy())
        info["held_auroc"] = mets.get("auroc")
        info["degenerate"] = bool(
            info.get("held_auroc") is not None
            and info["held_auroc"] < DEGENERATE_AUROC)
        if info["degenerate"]:
            print(f"    DEGENERATE held AUROC {info['held_auroc']:.3f} "
                  f"< {DEGENERATE_AUROC}; audit rows will still be written",
                  flush=True)
        with open(train_info_path, "w") as handle:
            json.dump(info, handle, indent=2)

    elapsed = round(time.time() - t0, 1)
    if loaded_checkpoint:
        audit_info = {
            "source_checkpoint": str(checkpoint),
            "stages": active_stages,
            "seconds": elapsed,
            "held_auroc": info.get("held_auroc"),
            "degenerate": info.get("degenerate"),
            "checkpoint_retained": cell == 0,
        }
        audit_path = artifact_path("audit", arm, model_name, fold=cell,
                                   ext="json", smoke=smoke)
        with open(audit_path, "w") as handle:
            json.dump(audit_info, handle, indent=2)
    else:
        info["seconds"] = elapsed
        with open(train_info_path, "w") as handle:
            json.dump(info, handle, indent=2)
    if cell != 0 and checkpoint.exists() and not keep_checkpoint:
        # Cell 0 is retained for later audits. Other checkpoints are only
        # interruption-safe scratch space and are removed after all outputs land
        # — unless a later audit needs the weights back (--keep-checkpoint),
        # which is cheaper than retraining the cell to score it again.
        shutil.rmtree(checkpoint)
    print(f"  {arm} fold{cell}: done in {elapsed}s  {info}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------- the fused SAE audit
# Formerly src/stage2/sae_audit.py: own-feature causal
# faithfulness rows for one freshly trained cell, written to
# results/stage2/ where the Stage 2 metric CLIs aggregate them.


def train_arm(method: str, model: str, cells, seed: int, allow: bool,
              keep_checkpoint: bool, *, stages=None, limit_eval=None,
              max_steps=None, force_retrain=False):
    """One FT arm (or the anchor): train per fold, score the held fold, and
    run the fused Stage 2 SAE audit before the checkpoint is released.
    `max_steps`/`limit_eval` mark a smoke run."""
    from src.baselines.fine_tune import ARMS
    from src.utils.data_prep import load_corpus
    from src.utils.data_prep import MODEL_REGISTRY
    from src.baselines.prompting import check_construct_hashes

    cfg = load_config()
    arm = ARM_OF[method]
    spec = ARMS[arm]
    stages = list(stages) if stages else list(ARM_STAGES)
    unknown = set(stages) - set(ARM_STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; use {ARM_STAGES}")
    require_model_process_approval(
        cfg, cli_override=allow, command="src.stage1.train")
    check_construct_hashes()

    if cells is None:
        cells = [0] if method == "anchor" else DEFAULT_FOLDS
    posts, _sentences = load_corpus("lonxplain")

    if max_steps is not None:
        cfg["full_finetune"]["min_steps"] = max_steps
        cfg["full_finetune"]["max_steps"] = max_steps
    smoke = any(value is not None for value in (max_steps, limit_eval))

    print(f"arm {arm} ({spec}) on {model}, folds {list(cells)}", flush=True)
    decision_namespace = "v3.smoke" if smoke else "v3.arm"
    record_decision(f"{decision_namespace}.{arm}.{model}",
                    {"spec": spec, "direction": "lonxplain",
                     "train_corpus": "lonxplain", "eval_corpus": None,
                     "split_by": "fold", "cells": list(cells),
                     "seed": seed, "recipe": cfg["full_finetune"],
                     "stages": stages, "smoke": smoke})
    for cell in cells:
        _run_cell(arm, spec, model, cell, cfg, seed, posts, stages,
                  limit_eval, force_retrain=force_retrain,
                  smoke=smoke, keep_checkpoint=keep_checkpoint)


# ------------------------------------------------------ reference detectors


def train_mentalbert(folds, seed: int, allow: bool,
                     encoder: str = MENTALBERT_ENCODER,
                     keep_checkpoint: bool = False):
    """The domain-pretrained encoder reference arm (not a fine-tuning cell)."""
    import json
    import time

    import numpy as np
    import pandas as pd

    from src.baselines import mental_bert
    from src.utils.data_prep import load_corpus
    from src.utils.common import best_f1_threshold, detection_metrics

    cfg = load_config()
    require_model_process_approval(cfg, cli_override=allow,
                                   command="src.stage1.train --method mentalbert")
    ecfg = cfg["encoder_finetune"]
    posts, _ = load_corpus("lonxplain")
    text_by_id = posts.set_index("id")["text"]
    label_by_id = posts.set_index("id")["label"]

    rows, aurocs = [], []
    print(f"MentalBERT reference arm "
          f"({mental_bert.MENTALBERT_REGISTRY[encoder]}), folds {folds}",
          flush=True)
    for fold in folds:
        t0 = time.time()
        train_ids = subsample(posts, fold, seed, ALL)
        y_tr = (label_by_id.loc[train_ids] == "present").astype(int).to_numpy()
        model, tok, info = mental_bert.train(
            encoder, [text_by_id.loc[p] for p in train_ids], y_tr, ecfg,
            seed=seed, log_every=200)
        max_len = info["max_len"]
        s_tr = mental_bert.predict(model, tok,
                                   [text_by_id.loc[p] for p in train_ids],
                                   ecfg, max_len)
        thr = best_f1_threshold(s_tr, y_tr)

        test = posts[posts["fold"] == fold]
        y_id = (test["label"] == "present").to_numpy().astype(int)
        s_id = mental_bert.predict(model, tok, test["text"].tolist(), ecfg, max_len)
        m = detection_metrics(y_id, (s_id >= thr).astype(int), s_id)
        aurocs.append(m["auroc"])
        rows.append(pd.DataFrame({
            "arm": "mentalbert_finetune", "model": encoder,
            "corpus": "lonxplain", "fold": fold, "id": test["id"].to_numpy(),
            "label": np.where(y_id == 1, "present", "absent"),
            "post_score": s_id,
            "post_pred": np.where(s_id >= thr, "present", "absent"),
        }))
        print(f"  fold {fold}: ID {m['auroc']:.4f}  "
              f"({round(time.time() - t0)}s)", flush=True)
        if keep_checkpoint:
            # Retained for the input-level attributions (src.stage2.occlusion):
            # the encoder's state dict plus what rebuilds the classifier.
            from src.utils.common import checkpoint_path

            ck = checkpoint_path("mentalbert", encoder, fold)
            ck.mkdir(parents=True, exist_ok=True)
            mental_bert.save_checkpoint(model, encoder, max_len, ck)
            print(f"    checkpoint retained at {ck.name}", flush=True)
        del model

    scores = artifact_path("detect", "mentalbert", encoder)
    scores.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(scores, index=False)
    summary = {"arm": "mentalbert_finetune", "model_id": info["model_id"],
               "params_m": info["params_m"], "recipe": ecfg, "folds": list(folds),
               "auroc_by_fold": [round(v, 4) for v in aurocs],
               "role": "detection reference arm — not one of the FT arms"}
    with open(scores.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    record_decision(f"v2.mentalbert_baseline.{encoder}", summary)


def train_glove_bigru(folds, keep_checkpoint: bool = False,
                      force_retrain: bool = False):
    """The corpus paper's own family (Garg et al. 2023): BiGRU over frozen
    GloVe, one training per fold at the full budget, seed 0. GPU (ungated: no
    study model is loaded), CPU when no card is visible."""
    import json
    import time

    import numpy as np
    import pandas as pd

    from src.baselines import glove_bigru as gb
    from src.utils.data_prep import load_corpus
    from src.utils.common import best_f1_threshold

    cfg = load_config()
    posts, _ = load_corpus("lonxplain")
    text_by_id = posts.set_index("id")["text"]
    label_by_id = posts.set_index("id")["label"]

    for fold in (folds if folds is not None else range(N_FOLDS)):
        pq = artifact_path("detect", "glove_bigru", DEFAULT_MODEL, fold=fold)
        js = artifact_path("train", "glove_bigru", DEFAULT_MODEL, fold=fold,
                           ext="json")
        if pq.exists() and js.exists() and not force_retrain:
            print(f"  fold {fold}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        train_ids = subsample(posts, fold, 0, ALL)
        y_tr = (label_by_id.loc[train_ids] == "present").astype(int).to_numpy()
        print(f"  fold {fold}: training glove_bigru on {len(train_ids)} posts",
              flush=True)
        model = gb.train([text_by_id.loc[p] for p in train_ids], y_tr, cfg, seed=0)
        thr = best_f1_threshold(
            gb.predict(model, [text_by_id.loc[p] for p in train_ids], cfg), y_tr)
        test = posts[posts["fold"] == fold]
        scores = gb.predict(model, test["text"].tolist(), cfg)
        if keep_checkpoint:
            from src.utils.common import checkpoint_path

            ck = checkpoint_path("glove_bigru", DEFAULT_MODEL, fold)
            ck.mkdir(parents=True, exist_ok=True)
            gb.save_checkpoint(model, cfg, ck)
        pq.parent.mkdir(parents=True, exist_ok=True)
        js.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "arm": "glove_bigru", "model": "glove-300d-bigru", "fold": fold,
            "id": test["id"].to_numpy(), "label": test["label"].to_numpy(),
            "post_score": scores, "word_score": np.nan,
            "post_pred": score_to_pred(scores, thr),
            "word_pred": None,
        }).to_parquet(pq, index=False)
        with open(js, "w") as f:
            json.dump({"arm": "glove_bigru", "fold": fold,
                       "n_train": len(train_ids), "threshold": float(thr),
                       "seconds": round(time.time() - t0, 1)}, f, indent=2)
        print(f"  fold {fold}: done in {round(time.time() - t0)}s", flush=True)

    # Assemble the whole-corpus parquet once all ten folds exist.
    pieces = []
    for fold in range(N_FOLDS):
        pq = artifact_path("detect", "glove_bigru", DEFAULT_MODEL, fold=fold)
        if not pq.exists():
            print(f"  not assembled — fold {fold} missing", flush=True)
            return
        pieces.append(pd.read_parquet(pq))
    out = artifact_path("detect", "glove_bigru", DEFAULT_MODEL)
    pd.concat(pieces, ignore_index=True).to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}")


def train_prompting(model: str, allow: bool):
    """The zero-shot prompting reference detector over all 3,522 posts (GPU)."""
    import json

    import pandas as pd

    from src.baselines.prompting import answer_prompts
    from src.baselines.prompting import load_llm
    from src.utils.data_prep import DATA
    from src.baselines.prompting import build_detection_prompt
    from src.baselines.prompting import check_construct_hashes

    cfg = load_config()
    require_model_process_approval(cfg, cli_override=allow,
                                   command="src.stage1.train --method prompting")
    check_construct_hashes()

    lonxplain = pd.read_csv(DATA / "lonxplain" / "posts.csv")
    jobs = [{"id": r["id"], "label": r["label"], "fold": int(r["fold"]),
             "text": r["text"],
             "prompt": build_detection_prompt(r["text"])}
            for _, r in lonxplain.iterrows()]

    print(f"prompting: {len(jobs)} prompts on {model}", flush=True)
    if is_api_model(model):
        # No backbone to load and no GPU to pin: the weights are behind an API.
        from src.baselines.prompting import answer_prompts_api

        answers = answer_prompts_api([j["prompt"] for j in jobs],
                                     [j["text"] for j in jobs], model)
    else:
        llm, tok = load_llm(model)
        answers = answer_prompts(llm, tok, [j["prompt"] for j in jobs],
                                 [j["text"] for j in jobs])
    out = pd.DataFrame({"id": [j["id"] for j in jobs],
                        "label": [j["label"] for j in jobs],
                        "fold": [j["fold"] for j in jobs], **answers})
    path = artifact_path("detect", "prompting", model, ext="csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    n_unparsed = int(out["pred_label"].isna().sum())
    kept = sum(len(json.loads(i)) for i in answers["evidence_word_indices"])
    discarded = sum(answers["discarded_words"])
    record_decision(f"prompting.{model}", {
        "model": model, "n_prompts": len(jobs), "unparsed_labels": n_unparsed,
        "unscored_posts": int(out["present_score"].isna().sum()),
        "evidence_word_discard_rate": (
            discarded / (kept + discarded) if (kept + discarded) else 0.0),
        "shot_scheme": "zero-shot; no labelled exemplars",
        "answer_contract": "reasoning, cause and consequence spans, active "
                           "dimensions, four rule flags, label",
    })
    print(f"wrote {path.name}  (unparsed: {n_unparsed})")


# -------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(
        description="Train post-level loneliness detectors on LonXplain.")
    add_method_args(ap)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="study folds (default 0 2 4 6 8; glove_bigru trains "
                         "all ten)")
    ap.add_argument("--seed", type=int, default=0)
    _dense = ", ".join(f"{m} {l}" for m, l in DENSE_PROBE_LAYER.items())
    # The SAE rows quote the dictionary's own layer first and the cached index
    # it lives at second: they are the same site, and printing only the cache
    # index reads as though the probe sits a block deeper than it does.
    _sae_pins = load_config().get("sae") or {}
    _sae = ", ".join(
        f"{m} {int(_sae_pins[m]['layer'])} (cached {l})"
        for m, l in SAE_PROBE_LAYER.items())
    ap.add_argument("--layer", type=int, default=None,
                    help="probes: cached layer override (defaults follow the "
                         f"SAE pin in configs/default.yaml — dense {_dense}; "
                         f"sae_probe {_sae})")
    ap.add_argument("--keep-checkpoint", action="store_true",
                    help="FT arms: retain non-fold-0 checkpoints for the "
                         "Stage 2 DLA audit")
    ap.add_argument("--force-retrain", action="store_true",
                    help="FT arms: retrain a cell whose outputs already exist, "
                         "archiving them to results/legacy_pre_v3/ first. This "
                         "is the path back to a discarded checkpoint: folds "
                         "2-8 keep their sidecar and their scores but not "
                         "their weights, so `--force-retrain "
                         "--keep-checkpoint` is how a Stage 2 metric gets a "
                         "cell it can score")
    ap.add_argument("--gpu", action="store_true",
                    help="let probe fitting use a GPU (default: pinned to CPU "
                         "so the published fits are reproduced exactly)")
    add_device_args(ap)
    ap.add_argument("--allow-model-process", action="store_true",
                    help="explicit study-owner approval to load a study model")
    args = ap.parse_args()

    methods = selected_methods(args)
    model = resolve_model(args.model)
    folds = args.folds if args.folds is not None else DEFAULT_FOLDS

    # Device selection happens before torch is imported, which is the last
    # moment CUDA_VISIBLE_DEVICES is still read. Probe fits stay pinned to CPU
    # unless a GPU method shares the invocation or --gpu asks for one.
    gpu_methods = set(methods) & GPU_METHODS
    if is_api_model(model):
        # An API backbone runs prompting and nothing else, and prompting on it
        # touches no card -- so it must not be held to a backbone's headroom.
        rest = sorted(set(methods) - {"prompting"})
        if rest:
            raise SystemExit(
                f"--model {model} is reached over the API: it has no "
                f"activations to cache and no weights to train, so it runs "
                f"prompting only. Drop {', '.join(rest)} or pick a local "
                f"backbone.")
        gpu_methods -= {"prompting"}
    needs_gpu = bool(args.gpu or gpu_methods)
    min_free_gb = (GLOVE_MIN_FREE_GB if set(methods) == {"glove_bigru"}
                   else GPU_MIN_FREE_GB.get(model, 24.0))

    pin_visible_devices(args.devices, need_gpu=needs_gpu,
                        min_free_gb=min_free_gb,
                        wait_minutes=args.wait_for_gpu)

    dense = [m for m in methods if m in DENSE_PROBES]
    if dense:
        train_dense_probes(dense, model, folds, args.seed, args.layer)
    if "sae_probe" in methods:
        train_sae_probe(model, folds, args.seed, args.layer,
                        args.allow_model_process)
    if "sae_attention_probe" in methods:
        train_sae_attention_probe(model, folds, args.seed, args.layer,
                                  args.allow_model_process)
    for method in methods:
        if method in ARM_OF:
            print(f"\n=== {method} ({ARM_OF[method]}) on {model} ===")
            train_arm(method, model, args.folds, args.seed,
                      args.allow_model_process, args.keep_checkpoint,
                      force_retrain=args.force_retrain)
    if "mentalbert" in methods:
        train_mentalbert(folds, args.seed, args.allow_model_process,
                         keep_checkpoint=args.keep_checkpoint)
    if "glove_bigru" in methods:
        train_glove_bigru(args.folds, keep_checkpoint=args.keep_checkpoint,
                          force_retrain=args.force_retrain)
    if "prompting" in methods:
        train_prompting(model, args.allow_model_process)


if __name__ == "__main__":
    sys.exit(main())
