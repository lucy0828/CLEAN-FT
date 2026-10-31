"""Shared plumbing for every stage CLI.

Paths & config, the decisions log, the model-process gate, the method
registry with the shared CLI options (--method, --model), the study's
fold/subsample machinery, and the detection metrics +
post bootstrap. Not run directly.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "activations"
# A seed REPLICATE writes everything it produces -- cells, checkpoints,
# sidecars and the published csvs -- under results/replicates/<tag>/ instead
# of results/, so a `--seed 1` run can never upsert its rows over the study's
# seed-0 tables. Set LONXPLAIN_REPLICATE=seed1 in the environment of every
# command of that replicate (train, detection, dla_auprc, word_deletion);
# `src.stage2.seed_variation` reads the replicates back beside seed 0.
REPLICATE = os.environ.get("LONXPLAIN_REPLICATE", "").strip()
_LAYER_VARIANT = os.environ.get("LONXPLAIN_DENSE_LAYER", "").strip()
if REPLICATE:
    RESULTS = ROOT / "results" / "replicates" / REPLICATE
elif _LAYER_VARIANT:
    # a read-out depth variant (see DENSE_LAYER_VARIANT below)
    RESULTS = (ROOT / "results" / "layer_variants"
               / _LAYER_VARIANT.replace(":", "_layer"))
else:
    RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# Where every generated file lives. One rule: a directory is named for what
# its files ARE, which is the stage prefix of their name (see `artifact` below),
# so a path and a file name can never disagree about what a file is. Use
# `artifact_path` rather than joining a directory yourself -- it picks both.
#
#   results/detection/    detect_*  — per-cell held-fold scores. Written by
#                         src.stage1.train, which computes no metric: it
#                         produces weights and the evidence to score them.
#   results/train/        train_*   — the recipe sidecar for each cell, and
#                         audit_*   — what a checkpoint-reusing rerun did.
#                         checkpoints/ holds the weights Stages 2 and 3 load.
#   results/stage2/       plaus_*, delete_* — the per-post
#                         intervention rows the Stage 2 metric CLIs compute,
#                         before they are aggregated into a table.
#   results/raw/stage3/   ood_*     — the clinical OOD deployment's
#                         per-phase scores, oodwords_* its per-word
#                         attributions and ooddelete_* its word-deletion
#                         rows, all written by src.stage3.ood_detection.
#   results/*.csv         the published metric tables, one per CLI.
#
# results/detection.csv is the published table; results/detection/ is the raw
# evidence it aggregates.
DETECTION_DIR = RESULTS / "detection"
TRAIN_DIR = RESULTS / "train"
STAGE2 = RESULTS / "stage2"
STAGE3 = RESULTS / "stage3"        # clinical OOD deployment on conversations
RAW_STAGE3 = RAW / "stage3"
CHECKPOINTS = TRAIN_DIR / "checkpoints"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------- artifact file names
#
# One shape for every generated file:
#
#     {stage}_{method}_{model}[_fold{k}].{ext}
#
# read left to right as "what produced it, for which method, on which backbone,
# from which fold". The stage prefix is what the file IS, not which CLI wrote
# it: training deposits DETECTION evidence, so those files are `detect_*` even
# though `src.stage1.train` writes them. The model is always named -- an
# omitted default suffix reads as "some model" the moment a second backbone
# exists, and this study has two.

METHOD_LABEL = {
    # fine-tuning arms, addressable by CLI method name or by internal arm name
    "full_ft": "ft",                     "finetune_full_random": "ft",
    "lp_ft": "lp_ft",                    "finetune_full_probe_init": "lp_ft",
    "attn_ft": "attn_ft",                "finetune_attn_random": "attn_ft",
    "attn_lp_ft": "attn_lp_ft",          "finetune_attn_probe_init": "attn_lp_ft",
    # the LoRA column: same 2x2, adapters instead of full weight updates
    "gxi_ft": "gxi_ft",                  "explain_gxi_full_random": "gxi_ft",
    "occl_attention_probe": "p_occl_attn",
    "occl_attn_ft": "occl_attn_ft",        "explain_occl_attn_random": "occl_attn_ft",
    "occl_midlayer_attn_ft": "occl_midlayer_attn_ft",
    "explain_occl_midlayer_attn_random": "occl_midlayer_attn_ft",
    "occl_midlayer_attn_lp_ft": "occl_midlayer_attn_lp_ft",
    "explain_occl_midlayer_attn_probe_init": "occl_midlayer_attn_lp_ft",
    "final_linear_probe": "p_linear_final",
    "final_attention_probe": "p_attn_final",
    "occl_attn_lp_ft": "occl_attn_lp_ft",  "explain_occl_attn_probe_init": "occl_attn_lp_ft",
    "occl_attn_lora_ft": "occl_attn_lora_ft",
    "explain_occl_lora_attn_random": "occl_attn_lora_ft",
    "occl_attn_span_ft": "occl_attn_span_ft",
    "explain_occl_span_attn_random": "occl_attn_span_ft",
    "occl_final_attention_probe": "p_occl_attn_final",
    "occl_last_token_probe": "p_occl_last",
    "occl_attention_mlp_probe": "p_occl_attn_mlp",
    "occl_sae_attention_probe": "p_occl_sae_attn",
    "occl_sae_probe": "p_occl_sae",
    "occl_mentalbert": "occl_mentalbert",  "explain_occl_mentalbert": "occl_mentalbert",
    "occl_glove_bigru": "occl_glove",      "explain_occl_glove_bigru": "occl_glove",
    "last_ft": "last_ft",                "finetune_last_random": "last_ft",
    "lora_ft": "lora_ft",                "finetune_lora_random": "lora_ft",
    "lora_lp_ft": "lora_lp_ft",          "finetune_lora_probe_init": "lora_lp_ft",
    "attn_lora_ft": "attn_lora_ft",      "finetune_lora_attn_random": "attn_lora_ft",
    "attn_lora_lp_ft": "attn_lora_lp_ft",
    "finetune_lora_attn_probe_init": "attn_lora_lp_ft",
    # span-supervised full fine-tuning: the expert span enters the loss
    "span_ft": "span_ft",                "finetune_span_random": "span_ft",
    "attn_span_ft": "attn_span_ft",      "finetune_span_attn_random": "attn_span_ft",
    "anchor": "anchor",
    # frozen probes: the p_ prefix keeps the family sorted together on disk
    "linear_probe": "p_linear",
    "attention_probe": "p_attn",
    "attention_mlp_probe": "p_attn_mlp",
    "sae_probe": "p_sae",
    "sae_attention_probe": "p_sae_attn",
    "mlp_probe": "p_mlp",
    "afr_probe": "p_afr",
    "last_token_probe": "p_last",
    # references
    "mentalbert": "mentalbert",
    "mentalbert_finetune": "mentalbert",
    "glove_bigru": "glove",
    "prompting": "prompt",
    # Not methods: labels for cell files that span several methods, which carry
    # a `method` column instead of one method per file.
    "cells": "cells",
    "probes": "probes",
}

MODEL_LABEL = {"gemma-2-2b": "gemma", "llama31-8b": "llama",
               "claude-opus-5": "opus"}

# mentalbert and glove_bigru load no study backbone (BACKBONE_FREE below), so
# --model does not name their file. mentalbert is filed under the encoder it
# fine-tunes; glove_bigru trains from scratch and is filed under the study
# default. Writer and reader both take the name from here.
MENTALBERT_ENCODER = "mentalbert"


def method_label(name: str) -> str:
    """Short on-disk label for a method or arm. Unknown names pass through."""
    return METHOD_LABEL.get(name, name)


def model_label(name: str) -> str:
    return MODEL_LABEL.get(name, name)


def artifact(stage: str, method: str, model: str, *, fold=None,
             ext: str = "parquet") -> str:
    """The canonical file name for one generated artifact.

    `fold` may be an int, or the string "*" for a glob -- callers that collect
    a whole arm's cells build the pattern through here too, so a rename cannot
    leave a writer and its reader disagreeing.
    """
    parts = [stage, method_label(method), model_label(model)]
    if fold is not None:
        parts.append(f"fold{fold}")
    return "_".join(parts) + f".{ext}"


# Which directory each stage prefix's files live in (see the layout note at the
# top of this module). audit_* sits with train_*: both record what a training
# run did, as opposed to what it produced.
STAGE_DIR = {
    "detect": DETECTION_DIR,
    "train": TRAIN_DIR,
    "audit": TRAIN_DIR,
    "plaus": STAGE2,
    "delete": STAGE2,
    "attrib": STAGE2,      # per-word own-vs-occlusion scores (src.stage2.occlusion)
    "eraser": STAGE2,      # ERASER sufficiency/comprehensiveness rows (same)
    "ood": RAW_STAGE3,
    "oodwords": RAW_STAGE3,
    "ooddelete": RAW_STAGE3,
}


def artifact_path(stage: str, method: str, model: str, *, fold=None,
                  ext: str = "parquet", smoke: bool = False):
    """Full path for one generated artifact -- its directory and its name.

    Both come from the stage prefix, so a writer and a reader that ask for the
    same artifact cannot end up in different directories. `fold="*"` builds a
    glob, exactly as in `artifact`. `smoke` mirrors the layout one level down,
    where debug runs land so they can never be mistaken for study cells.
    """
    directory = STAGE_DIR[stage]
    if smoke:
        directory = directory / "smoke"
    return directory / artifact(stage, method, model, fold=fold, ext=ext)


def checkpoint_path(arm: str, model: str, cell: int, *, smoke: bool = False):
    """Where one trained fine-tuning cell's weights live (fold 0 is retained)."""
    directory = CHECKPOINTS / "smoke" if smoke else CHECKPOINTS
    return directory / f"{method_label(arm)}_{model_label(model)}_fold{cell}"


def record_decision(key: str, value, path: Path | None = None):
    """Append a decision to results/decisions.json (created if absent).

    Safe under concurrent writers (folds are parallelized across GPUs): the
    read-modify-write is serialized by an advisory lock and the replace is
    atomic, so concurrent runs cannot lose or corrupt entries.
    """
    import fcntl
    import os

    path = path or (RESULTS / "decisions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".json.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        decisions = {}
        if path.exists():
            with open(path) as f:
                decisions = json.load(f)
        decisions[key] = value
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(decisions, f, indent=2, sort_keys=True)
        os.replace(tmp, path)


def load_dotenv() -> None:
    """Read ANTHROPIC_* keys out of a repo-root .env, if one is there.

    The SDK resolves credentials from the environment, and this repo keeps its
    key in a gitignored .env rather than in the shell profile, so the value is
    never exported into the shell every GPU job inherits. Anything already set
    in the environment wins -- this only fills gaps.
    """
    import os

    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key.startswith("ANTHROPIC_") and not os.environ.get(key):
            os.environ[key] = value


def require_model_process_approval(
    config: dict,
    *,
    cli_override: bool,
    command: str,
) -> None:
    """Refuse model work unless config or an explicit CLI override allows it."""
    configured = bool(config.get("execution", {}).get("allow_model_process", False))
    if not (configured or cli_override):
        raise RuntimeError(
            f"{command} would load/train a study model, but "
            "execution.allow_model_process=false. Review the v3 code first, "
            "then pass --allow-model-process only after explicit approval."
        )


# ------------------------------------------------------------------ devices
#
# Every GPU method here is written for a single card (configs/default.yaml pins
# 8-bit Adam so that even an 8B backbone fits one 80GB A100), and the training
# code says `.cuda()`, which is always *visible* device 0. On a shared box that
# is whichever card the driver enumerates first -- usually the one somebody else
# is already filling. Choosing the visible set before torch is imported keeps
# the single-card recipe intact and simply points it at a free card.


def gpu_free_memory() -> list[tuple[int, int]]:
    """[(physical index, free MiB)] from nvidia-smi, freest first.

    Read through nvidia-smi rather than torch so this can run before CUDA is
    initialized -- CUDA_VISIBLE_DEVICES is only honoured up to that point.
    Returns [] when there is no usable driver.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in out.strip().splitlines():
        index, _, free = line.partition(",")
        try:
            devices.append((int(index), int(free)))
        except ValueError:
            continue
    return sorted(devices, key=lambda d: -d[1])


# What one card must have free for a run on each backbone: bf16 weights plus
# room for the activations inference produces. A floor, not a measurement --
# it exists so `auto` refuses a half-full card instead of selecting it and
# dying inside the first forward pass. Training passes its own larger number.
GPU_MIN_FREE_GB = {"gemma-2-2b": 12.0, "llama31-8b": 30.0}


def min_free_gb_for(model: str | None, default: float = 12.0) -> float:
    """The free-memory floor one inference process on `model` needs."""
    return GPU_MIN_FREE_GB.get(model, default)


def pin_visible_devices(spec: str, *, need_gpu: bool, min_free_gb: float = 20.0,
                        count: int = 1, wait_minutes: float = 0.0,
                        poll_seconds: float = 30.0) -> str:
    """Set CUDA_VISIBLE_DEVICES for this process and return what it was set to.

    spec is a CLI value: "cpu" hides every GPU, "auto" takes the `count`
    emptiest cards with at least `min_free_gb` free, "all" leaves the driver
    order alone, and anything else is a literal device list ("1,2").
    Must be called before torch is imported. An externally set
    CUDA_VISIBLE_DEVICES always wins -- the caller asked for those cards.

    `wait_minutes` > 0 makes "auto" poll instead of giving up, which is what a
    shared box wants: the alternative is a human re-running the command until
    a neighbour's job ends.
    """
    if not need_gpu or spec == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return ""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        print(f"GPU: using inherited CUDA_VISIBLE_DEVICES={inherited}")
        return inherited
    if spec == "all":
        return ""
    if spec != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = spec
        print(f"GPU: CUDA_VISIBLE_DEVICES={spec} (requested)")
        return spec
    import time

    deadline = time.monotonic() + max(0.0, wait_minutes) * 60.0
    announced = False
    while True:
        devices = gpu_free_memory()
        if not devices:
            print("GPU: nvidia-smi reported no devices; leaving allocation to "
                  "torch")
            return ""
        free_enough = [d for d in devices if d[1] >= min_free_gb * 1024]
        if free_enough:
            break
        best, free = devices[0]
        shortfall = (f"no GPU has the {min_free_gb:.0f} GB this run needs "
                     f"(emptiest is cuda:{best} with {free / 1024:.1f} GB free)")
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"{shortfall}. Wait for a card (--wait-for-gpu MINUTES), pass "
                "--devices N to use one anyway, or --devices cpu.")
        if not announced:
            print(f"GPU: {shortfall}; waiting up to {wait_minutes:g} min "
                  f"(polling every {poll_seconds:g}s)", flush=True)
            announced = True
        time.sleep(poll_seconds)
    chosen = [str(index) for index, _ in free_enough[:count]]
    value = ",".join(chosen)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    report = ", ".join(f"cuda:{i} {f / 1024:.0f} GB free"
                       for i, f in free_enough[:count])
    print(f"GPU: CUDA_VISIBLE_DEVICES={value} (auto-selected: {report})")
    return value


def add_device_args(ap: argparse.ArgumentParser) -> None:
    """The shared --devices / --wait-for-gpu pair."""
    ap.add_argument("--devices", default="auto", metavar="SPEC",
                    help="which GPUs this process may see: auto (default: the "
                         "emptiest free card), all, cpu, or an explicit list "
                         "like 1,2. An existing CUDA_VISIBLE_DEVICES wins. "
                         "One process uses ONE card; to use several, run "
                         "several processes, each pinned to its own card "
                         "(--devices 1, --devices 2, ... over --folds).")
    ap.add_argument("--wait-for-gpu", type=float, default=0.0,
                    metavar="MINUTES", dest="wait_for_gpu",
                    help="--devices auto: poll this long for a card with "
                         "enough free memory instead of exiting (default 0)")


def cuda_or_exit(module, *, what: str):
    """`module.cuda()`, or a message naming the cards instead of a torch dump.

    Every GPU method here is a single-card recipe, so `.cuda()` is visible
    device 0 -- whatever `pin_visible_devices` chose. It can still fill up
    between selection and use on a shared box, and the raw OutOfMemoryError
    says nothing about which card that is or what to do next.
    """
    import torch

    try:
        return module.cuda()
    except torch.OutOfMemoryError as exc:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
        cards = ", ".join(f"cuda:{i} {free / 1024:.1f} GB free"
                          for i, free in gpu_free_memory()) or "none reported"
        raise SystemExit(
            f"out of memory loading {what} on visible device 0 "
            f"(CUDA_VISIBLE_DEVICES={visible}).\n"
            f"  cards now: {cards}\n"
            "  rerun with --devices N to pin a free card, --wait-for-gpu "
            "MINUTES to queue for one, or --devices cpu.") from exc


# Frozen probes: fit on cached activations, backbone never updated.
PROBES = ("linear_probe", "mlp_probe", "attention_probe",
          "shrunk_attention_probe",
          "attention_mlp_probe", "sae_probe", "sae_attention_probe",
          "afr_probe", "last_token_probe",
          # a post-hoc explainer over a probe files with the probes: its
          # decisions ARE linear_probe's (EXPLAINER_BASE)
          "occl_attention_probe",
          "occl_final_attention_probe", "occl_last_token_probe",
          "occl_attention_mlp_probe", "occl_sae_attention_probe", "occl_sae_probe",
          # the FINAL-layer mean-pooled linear probe (2026-09-09): the
          # depth-matched frozen control against Full FT, fitted on the
          # cache's final-layer file (FINAL_PROBE_LAYER)
          "final_linear_probe", "final_attention_probe")

# The probes fitted from the cached dense residual stream by
# `src.probe.FITTERS`. The SAE arms are excluded because they load a dictionary
# and take --allow-model-process; every caller that has to split "dense probes"
# from "SAE probes" reads this tuple. It was written out by hand in
# src/stage1/train.py and again in src/stage2/dla_auprc.py until 2026-08-28,
# and the second copy is exactly the kind of drift dla_auprc's own header
# warns about.
DENSE_PROBES = ("linear_probe", "mlp_probe", "attention_probe",
                "shrunk_attention_probe",
                "attention_mlp_probe", "afr_probe", "last_token_probe",
                "final_linear_probe", "final_attention_probe")

# Method -> the variants its published rows are split by. Empty: every method
# in the study publishes exactly one row per (method, model), so the `variant`
# column is empty everywhere. Kept as the one place a method that DOES split
# its rows -- the way `sae_probe` would if it published one pooling per row --
# would declare it, rather than each CLI growing its own special case.
VARIANTS_OF = {}


def variants_of(method: str) -> tuple[str, ...]:
    """The variants `method` publishes a row for; `("",)` when it has none."""
    return VARIANTS_OF.get(method, ("",))

# Trained / prompted detectors. The four fine-tuning arms of the pooling x
# head-initialization 2x2, the anchor, and the three reference systems.
BASELINES = ("full_ft", "lp_ft", "attn_ft", "attn_lp_ft",
             "midlayer_ft", "midlayer_attn_ft",
             "midlayer_lp_ft", "midlayer_attn_lp_ft",
             "lora_ft", "lora_lp_ft", "attn_lora_ft", "attn_lora_lp_ft",
             "span_ft", "attn_span_ft", "gxi_ft",
             "occl_attn_ft", "occl_midlayer_attn_ft", "occl_midlayer_attn_lp_ft",
             "occl_attn_lp_ft", "occl_mentalbert", "occl_glove_bigru",
             "occl_attn_lora_ft", "occl_attn_span_ft",
             "last_ft", "anchor",
             "mentalbert", "glove_bigru", "prompting")

ALL_METHODS = PROBES + BASELINES

# Readable name -> the arm name raw cells are stored under.
ARM_OF = {
    "full_ft": "finetune_full_random",
    "lp_ft": "finetune_full_probe_init",
    # The attention-pooled column of the 2x2: same recipe, learned query
    # instead of the mean (fine_tune.AttentionPooledClassifier). attn_lp_ft is
    # the AP-LP-FT arm -- attention-pooled probe solved first, then fine-tuned.
    "attn_ft": "finetune_attn_random",
    "attn_lp_ft": "finetune_attn_probe_init",
    # The mid-layer-headed column: the head attached at DENSE_PROBE_LAYER
    # instead of at the final layer, so each cell sits on the site its frozen
    # probe partner reads. See fine_tune.ARMS for why. Only the PROBE-INIT
    # cells are kept -- the random-head pair was removed on 2026-08-28, its
    # artifacts quarantined to ~/lonxplain_removed_midlayer_random_20260828.
    "midlayer_ft": "finetune_midlayer_random",
    "midlayer_attn_ft": "finetune_midlayer_attn_random",
    "midlayer_lp_ft": "finetune_midlayer_probe_init",
    "midlayer_attn_lp_ft": "finetune_midlayer_attn_probe_init",
    # The LoRA column (2026-09-08): the pooling x head-initialization 2x2
    # again, with low-rank adapters on the attention and MLP projections
    # instead of full weight updates (fine_tune.ARMS, recipe="lora"). The
    # adapters are merged into the backbone before the checkpoint is written,
    # so every Stage 2/3 consumer reads these cells exactly as it reads the
    # full fine-tunes.
    "lora_ft": "finetune_lora_random",
    "lora_lp_ft": "finetune_lora_probe_init",
    "attn_lora_ft": "finetune_lora_attn_random",
    "attn_lora_lp_ft": "finetune_lora_attn_probe_init",
    # Span-supervised full fine-tuning (2026-09-08): full_ft / attn_ft with
    # the expert span added to the loss as a token-level target on the head's
    # own projection w . h_t (fine_tune.ARMS, objective="span"). Stage 2 then
    # scores these cells on posts whose spans they were trained on the
    # *other* folds of, exactly as the post label is out of fold.
    "span_ft": "finetune_span_random",
    "attn_span_ft": "finetune_span_attn_random",
    # Post-hoc explainers (2026-09-09): NOT trained arms. They share a base
    # arm's detection cells (EXPLAINER_BASE) and write their own plaus_* /
    # delete_* cells under their arm name (CELLS_FROM).
    # Gradient x input (2026-09-09) over the same Full FT checkpoint: the
    # gradient-based post-hoc explainer, through the backbone to the input
    # embeddings (src.stage2.gradient_input).
    "gxi_ft": "explain_gxi_full_random",
    # Occlusion (leave-one-word-out, 2026-09-09): the one uniform input-level
    # attribution run on the four principal read-outs, beside each one's own
    # decomposition (src.stage2.occlusion, which also reports their rank
    # agreement and ERASER sufficiency in probability space).
    "occl_attn_ft": "explain_occl_attn_random",
    "occl_midlayer_attn_ft": "explain_occl_midlayer_attn_random",
    "occl_midlayer_attn_lp_ft": "explain_occl_midlayer_attn_probe_init",
    # the two reference detectors under occlusion (text-only black boxes);
    # their cells are FILED under the study default backbone label because
    # the aggregators key on a backbone -- the value is backbone-free
    "occl_mentalbert": "explain_occl_mentalbert",
    "occl_glove_bigru": "explain_occl_glove_bigru",
    "occl_attn_lp_ft": "explain_occl_attn_probe_init",
    # the attention-pooled LoRA and span-supervised fine-tunes under
    # occlusion (2026-09-10), for the main table's occlusion column
    "occl_attn_lora_ft": "explain_occl_lora_attn_random",
    "occl_attn_span_ft": "explain_occl_span_attn_random",
    # Last-token read-out fine-tune (2026-09-09). Its own decomposition is a
    # one-hot, so its Stage 2 cells are gradient x input as well.
    "last_ft": "finetune_last_random",
    "anchor": "anchor",
}
METHOD_OF_ARM = {arm: name for name, arm in ARM_OF.items()}

# Explainer method -> the trained arm whose checkpoint it explains. An
# explainer has no checkpoint of its own: `src.stage1.detection` reads the
# base arm's decisions for it, and the `--compute` paths of the Stage 2 CLIs
# skip it (its cells come from the module named in CELLS_FROM).
EXPLAINER_BASE = {"gxi_ft": "full_ft",
                  "occl_attn_ft": "attn_ft",
                  "occl_midlayer_attn_ft": "midlayer_attn_ft",
                  "occl_midlayer_attn_lp_ft": "midlayer_attn_lp_ft",
                  "occl_attention_probe": "attention_probe",
                  "occl_attn_lp_ft": "attn_lp_ft",
                  "occl_attn_lora_ft": "attn_lora_ft",
                  "occl_attn_span_ft": "attn_span_ft",
                  "occl_final_attention_probe": "final_attention_probe",
                  "occl_last_token_probe": "last_token_probe",
                  "occl_attention_mlp_probe": "attention_mlp_probe",
                  "occl_sae_attention_probe": "sae_attention_probe",
                  "occl_sae_probe": "sae_probe",
                  "occl_mentalbert": "mentalbert",
                  "occl_glove_bigru": "glove_bigru"}

# Methods whose Stage 2 cells (plaus_* / delete_*) are written by a module of
# their own rather than by the `--compute` path of src.stage2.dla_auprc and
# src.stage2.word_deletion: the post-hoc explainers, and the last-token
# read-outs, whose own decomposition is a one-hot and so is not a ranking.
# The two aggregators skip these in --compute and read their cells as-is.
CELLS_FROM = {
    "occl_attn_ft": "src.stage2.occlusion",
    "occl_midlayer_attn_ft": "src.stage2.occlusion",
    "occl_midlayer_attn_lp_ft": "src.stage2.occlusion",
    "occl_attention_probe": "src.stage2.occlusion",
    "occl_attn_lp_ft": "src.stage2.occlusion",
    "occl_attn_lora_ft": "src.stage2.occlusion",
    "occl_attn_span_ft": "src.stage2.occlusion",
    "occl_final_attention_probe": "src.stage2.occlusion",
    "occl_last_token_probe": "src.stage2.occlusion",
    "occl_attention_mlp_probe": "src.stage2.occlusion",
    "occl_sae_attention_probe": "src.stage2.occlusion",
    "occl_sae_probe": "src.stage2.occlusion",
    "occl_mentalbert": "src.stage2.occlusion",
    "occl_glove_bigru": "src.stage2.occlusion",
    "gxi_ft": "src.stage2.gradient_input",
    # The two reference detectors' OWN decomposition (2026-09-10): both pool
    # by the mean and read with a linear head, so DLA is exact for them too;
    # src.stage2.reference_dla writes their plaus_/delete_ cells under the
    # study default backbone label, as their occlusion cells are.
    "mentalbert": "src.stage2.reference_dla",
    "glove_bigru": "src.stage2.reference_dla",
    "last_ft": "src.stage2.gradient_input",
    "last_token_probe": "src.stage2.gradient_input",
}

MODEL_ALIASES = {
    "gemma2": "gemma-2-2b", "gemma": "gemma-2-2b", "gemma-2-2b": "gemma-2-2b",
    "llama": "llama31-8b", "llama3": "llama31-8b", "llama31-8b": "llama31-8b",
    "opus": "claude-opus-5", "claude-opus-5": "claude-opus-5",
}

# Backbones reached over the Anthropic API rather than loaded locally. They are
# weights this study cannot open: no activations to cache, no gradients to take
# and -- the constraint that shapes what they can publish -- no token
# distribution to read a score off. An API backbone can therefore run
# `prompting` and nothing else, and the row it produces carries a decision with
# no AUROC beside it (src/stage1/METRICS.md).
API_MODELS = ("claude-opus-5",)


def is_api_model(model: str) -> bool:
    return model in API_MODELS

# Pre-registered cached layer per backbone for the dense probes: the 0.65-depth
# working layer.
DENSE_PROBE_LAYER = {"gemma-2-2b": 17, "llama31-8b": 24}

# A read-out depth VARIANT (2026-10-06, reviewer response). The depth sweep
# (src.stage1.layer_sweep) locates the layer where the frozen attention probe
# peaks; to train the mid-layer arms at that layer without touching the
# published tree, set LONXPLAIN_DENSE_LAYER="<model>:<layer>" in the
# environment of every command of the variant (train, detection, dla_auprc,
# word_deletion, uncertainty). It moves DENSE_PROBE_LAYER for that backbone
# and relocates RESULTS (cells, checkpoints, csvs) to
# results/layer_variants/<model>_layer<N>/, exactly as LONXPLAIN_REPLICATE
# relocates a seed replicate. The two cannot be combined.
DENSE_LAYER_VARIANT = os.environ.get("LONXPLAIN_DENSE_LAYER", "").strip()
if DENSE_LAYER_VARIANT:
    _vm, _vl = DENSE_LAYER_VARIANT.split(":")
    if _vm not in DENSE_PROBE_LAYER:
        raise SystemExit(f"LONXPLAIN_DENSE_LAYER names unknown backbone {_vm!r}")
    if REPLICATE:
        raise SystemExit("LONXPLAIN_DENSE_LAYER and LONXPLAIN_REPLICATE cannot "
                         "be combined")
    DENSE_PROBE_LAYER[_vm] = int(_vl)

# The cache's FINAL-layer file (hidden_states[num_layers]: the last block's
# output, pre final-norm), encoded with `data_prep --encode --fractions 1.0`.
# Only `final_linear_probe` reads it: the frozen control at the depth Full FT's
# head reads, against the mid-layer probes at the depth ML-FT's head reads.
FINAL_PROBE_LAYER = {"gemma-2-2b": 26, "llama31-8b": 32}
FINAL_LAYER_PROBES = ("final_linear_probe", "final_attention_probe")


def probe_layer_of(method: str, model: str, default: int | None = None) -> int:
    """The cached layer a dense probe is fitted on: DENSE_PROBE_LAYER for every
    probe except the final-layer ones."""
    if method in FINAL_LAYER_PROBES:
        return FINAL_PROBE_LAYER[model]
    return DENSE_PROBE_LAYER[model] if default is None else default

# hidden_states[k] is the *output* of block k-1, so a dictionary trained at
# blocks.N.hook_resid_post is cached one index above its own config layer.
# Confirmed per pin by src.probe.sae_probe.reconstruction_diagnostics
# (reconstruction error is minimal at the SAE's true site): Llama Scope 23 ->
# cached 24, Gemma Scope 12 -> cached 13.
SAE_CACHE_OFFSET = 1


def sae_cache_layer(sae_config: dict) -> int:
    """The cached layer the pinned dictionary actually reconstructs."""
    return int(sae_config["layer"]) + SAE_CACHE_OFFSET


def _sae_probe_layers() -> dict[str, int]:
    """sae_probe's cached layer per backbone, read off the SAE pin.

    Derived rather than written down a second time: a hardcoded copy drifts
    silently when the dictionary is repinned, and fitting the probe on the
    wrong cached layer raises nothing -- it just returns different numbers
    under the same name.
    """
    return {model: sae_cache_layer(pin)
            for model, pin in (load_config().get("sae") or {}).items()}


SAE_PROBE_LAYER = _sae_probe_layers()


def resolve_model(name: str | None) -> str:
    """Canonical backbone key. `None` (the --model default) means DEFAULT_MODEL."""
    if name is None:
        return DEFAULT_MODEL
    try:
        return MODEL_ALIASES[name.lower()]
    except KeyError:
        raise SystemExit(
            f"unknown --model {name!r}; use gemma2, llama or opus") from None


def group_of(method: str) -> str:
    return "probe" if method in PROBES or method == "frozen_backbone" else "baseline"


# glove_bigru trains a small net from scratch and mentalbert fine-tunes its own
# encoder, so neither reads a study backbone: their trainers take no model
# argument and their detection loaders discard it and label their own rows.
BACKBONE_FREE = ("glove_bigru", "mentalbert")


def add_method_args(ap: argparse.ArgumentParser, choices=ALL_METHODS):
    """The shared selection flags: --method and --model, both optional.

    --model defaults to DEFAULT_MODEL and is ignored outright by the
    BACKBONE_FREE methods, so `--method glove_bigru` needs no backbone at all.
    """
    ap.add_argument("--method", nargs="+", choices=choices, metavar="NAME",
                    default=None,
                    help="methods to run (default: all of " + ", ".join(choices)
                         + ")")
    ap.add_argument("--model", default=None,
                    help=f"backbone: gemma2, llama, or opus for Claude Opus 5 "
                         f"over the API, which runs prompting only "
                         f"(default {DEFAULT_MODEL}; "
                         f"{' and '.join(BACKBONE_FREE)} ignore it entirely)")


def selected_methods(args, choices=ALL_METHODS) -> list[str]:
    if getattr(args, "method", None):
        return list(dict.fromkeys(args.method))
    return list(choices)


DEFAULT_MODEL = "gemma-2-2b"
N_FOLDS = 10


def score_to_pred(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, "present", "absent")


def span_eval_ids(posts, cfg, fold: int, limit: int | None = None) -> list[str]:
    """The preregistered span-eval subset for one fold.

    LonXplain is the only corpus with expert spans: up to
    `audit.eval_posts_per_fold` present posts with an aligned span, drawn with
    a fold-pinned seed so every audit scores the identical posts.
    """
    import numpy as np

    pool = posts[
        (posts["fold"] == fold)
        & (posts["label"] == "present")
        & (posts["span_start"] >= 0)
    ]["id"].to_numpy()
    count = int(cfg["audit"]["eval_posts_per_fold"])
    if len(pool) <= count:
        ids = sorted(pool.tolist())
    else:
        rng = np.random.default_rng(9_000 + fold)
        ids = sorted(pool[rng.choice(len(pool), size=count, replace=False)].tolist())
    return ids[:limit] if limit else ids


# ------------------------------------------------- training subsamples
# Every trained arm at a cell (fold, seed) trains on the *identical* posts in
# the *identical* stratified order: membership and order are a pure function
# of (fold, seed) and the corpus.

ALL = -1  # budget sentinel meaning "every train-fold post"


def nested_order(posts: pd.DataFrame, fold: int, seed: int) -> list[str]:
    """A stratified ordering of fold `fold`'s training posts.

    Every prefix is class-balanced to the training pool's own rate, and longer
    budgets extend shorter ones rather than replacing them.
    """
    train = posts[posts["fold"] != fold]
    rng = np.random.default_rng((seed + 1) * 100_003 + fold)

    pres = train[train["label"] == "present"]["id"].to_numpy()
    absn = train[train["label"] == "absent"]["id"].to_numpy()
    pres = pres[rng.permutation(len(pres))]
    absn = absn[rng.permutation(len(absn))]

    p = len(pres) / (len(pres) + len(absn))
    order, i_pres, i_abs = [], 0, 0
    for k in range(1, len(pres) + len(absn) + 1):
        want_present = i_pres < round(p * k)
        if (want_present and i_pres < len(pres)) or i_abs >= len(absn):
            order.append(pres[i_pres])
            i_pres += 1
        else:
            order.append(absn[i_abs])
            i_abs += 1
    return order


def subsample(posts: pd.DataFrame, fold: int, seed: int, n: int) -> list[str]:
    """The `n` training post ids of cell (fold, seed); n=ALL means all of them."""
    order = nested_order(posts, fold, seed)
    return order if n == ALL else order[:n]


# ======================================================= metrics & bootstrap

# ------------------------------------------------------------ fitting helpers


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing present-class F1 (labels: 1=present)."""
    order = np.argsort(scores)
    candidates = np.r_[scores[order], scores.max() + 1.0]
    best_t, best_f1 = candidates[0] - 1.0, -1.0
    for t in np.unique(candidates):
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# ------------------------------------------------------------------ detection


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores=None) -> dict:
    """y_true/y_pred: 1=present, 0=absent. scores: continuous, for AUROC."""
    out = {}
    for cls, name in [(1, "present"), (0, "absent")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f"{name}_precision"] = p
        out[f"{name}_recall"] = r
        out[f"{name}_f1"] = f
    out["accuracy"] = float((y_pred == y_true).mean())
    if scores is not None and len(np.unique(y_true)) == 2:
        finite = np.isfinite(scores)
        out["auroc"] = float(roc_auc_score(y_true[finite], scores[finite]))
    return out


# ----------------------------------------------------------------- bootstrap
# One seeded draw matrix, reused by every method so cross-method contrasts
# stay paired.

def shared_draws(n: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """One resampled index matrix, reused by every method."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def metric_distribution(metric, y: np.ndarray, s: np.ndarray,
                        draws: np.ndarray) -> np.ndarray:
    """`metric(y, s)` on each resample; NaN where a resample is single-class."""
    out = np.full(len(draws), np.nan)
    for i, idx in enumerate(draws):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        out[i] = metric(yy, s[idx])
    return out


def ci(dist: np.ndarray) -> tuple[float, float]:
    """[2.5, 97.5] percentiles of a bootstrap distribution; NaN when empty."""
    if not np.isfinite(dist).any():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(values, n_boot: int = 2000, seed: int = 0):
    """(mean, lo, hi) of a per-post statistic under an ordinary post bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = [values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)
