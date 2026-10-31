"""Causal faithfulness by word deletion -> results/word_deletion.csv.

How to run (CPU, aggregates raw audit cells; --compute re-encodes the probe cells, which needs a GPU).
--method defaults to every method, --model defaults to gemma2:
    uv run python -m src.stage2.word_deletion                  # all methods, gemma2
    uv run python -m src.stage2.word_deletion --model llama
    uv run python -m src.stage2.word_deletion --method linear_probe attention_probe
    uv run python -m src.stage2.word_deletion --method full_ft lp_ft anchor
    uv run python -m src.stage2.word_deletion --compute --allow-model-process

The metric: delete the words the method's OWN word ranking selects from the text,
re-encode, and measure how much of the clean-to-empty logit gap disappears
(normalized AOPC over the k grid). Deletion is word-native but off-manifold --
a shortened post is text the model never saw -- so target and control are
matched in count and the published number is their difference, which pays that
cost on both sides.

The reported number is the **margin**: target minus a random-word control
matched in count. Raw source (results/stage2/): the fine-tuning arms'
delete_{method}_{model}_fold{k}.parquet and the probe cells
delete_probes_{model}.parquet, both written by `--compute`.
"""

import argparse
import glob as globlib

import numpy as np
import pandas as pd

from src.utils.common import artifact, mean_ci
from src.utils.common import (ARM_OF, CELLS_FROM, METHOD_OF_ARM, RESULTS,
                              add_device_args,
                              artifact_path, add_method_args, group_of,
                              min_free_gb_for, pin_visible_devices,
                              resolve_model, selected_methods)


def word_scores_from_tokens(fitted, ctx, pid, reduction: str) -> dict:
    """{word index: the method's own score for that word}.

    Sub-word tokens are reduced per word with the pinned aggregation, the same
    reduction `src.stage2.dla_auprc` uses, so a word is ranked here exactly as
    it is ranked there.
    """
    from src.probe import word_targets

    row = ctx.posts.set_index("id").loc[pid]
    targets = word_targets(np.asarray(ctx.acts[pid]["offsets"]), row["text"],
                           int(row["span_start"]), int(row["span_end"]))
    scores = np.asarray(fitted.token_scores(ctx.index_of[pid]), dtype=np.float64)
    out: dict[int, list[float]] = {}
    for t, w in zip(targets.alignment.token_indices,
                    targets.alignment.word_indices):
        if int(t) < len(scores):
            out.setdefault(int(w), []).append(float(scores[int(t)]))
    agg = {"max": max, "mean": lambda v: float(np.mean(v)), "sum": sum}[reduction]
    return {w: agg(v) for w, v in out.items()}


# ============================================================ the metric
INTERVENTION = "word_delete"
CONTROL = "random_words"
OUT_CSV = RESULTS / "word_deletion.csv"
# `afr_probe` and `anchor` were dropped from this metric on 2026-08-28: the
# faithfulness table now carries exactly the method set results/dla_auprc.csv
# does, so a method's plausibility and its self-faithfulness are always
# readable on the same axis. Both still exist elsewhere in the study.
#
# `sae_probe` and `sae_attention_probe` used to be excluded here because their
# readout is not a function of the residual stream -- they fit on SAE LATENTS at
# SAE_PROBE_LAYER while this audit re-encodes deleted text at DENSE_PROBE_LAYER
# and scores it with `probing.FITTERS`, which has no fitter for them. They are
# included since 2026-08-28 through their own path
# (`src.utils.sae_frozen_probe`), which re-encodes at SAE_PROBE_LAYER and pushes
# the fresh vectors through the pinned dictionary before the readout. That is
# the SAE as a study ARM scored on its OWN ranking; it is unrelated to the
# clinical-SAE-RANKED metric removed the same day.
SAE_METHODS = ("sae_probe", "sae_attention_probe")
METHODS = ("linear_probe", "mlp_probe", "attention_probe",
           "attention_mlp_probe", *SAE_METHODS,
           "full_ft", "lp_ft", "attn_ft", "attn_lp_ft",
           "midlayer_ft", "midlayer_attn_ft",
           "midlayer_lp_ft",
           "midlayer_attn_lp_ft",
           # The LoRA column and the span-supervised pair (2026-09-08): merged
           # checkpoints, so they are scored through the same arm path.
           "lora_ft", "lora_lp_ft", "attn_lora_ft", "attn_lora_lp_ft",
           "span_ft", "attn_span_ft",
           # Gradient x input over Full FT and the two last-token read-outs
           # (2026-09-09): cells from src.stage2.gradient_input.
           "gxi_ft", "last_ft", "last_token_probe",
           "occl_attn_ft", "occl_midlayer_attn_ft", "occl_midlayer_attn_lp_ft",
           "occl_attention_probe", "final_linear_probe", "final_attention_probe",
           "occl_attn_lp_ft", "occl_attn_lora_ft", "occl_attn_span_ft",
           "occl_final_attention_probe",
           "occl_last_token_probe", "occl_attention_mlp_probe",
           "occl_sae_attention_probe", "occl_sae_probe", "occl_mentalbert",
           "occl_glove_bigru",
           # the two references' own decomposition (src.stage2.reference_dla)
           "mentalbert", "glove_bigru")


def _margins(frame, keys, *, positive_gap_only):
    """The published aggregation, inlined for this intervention.

    Per post: target = MAX normalized AOPC over the k rows, control = MEAN
    over its k rows; margin = mean per-post difference with a 2,000-draw
    seed-0 post bootstrap.

    `positive_gap_only` drops posts whose clean-to-empty logit gap is not
    positive. That gap is the denominator of `normalized_decay`, so a
    non-positive one flips the sign of every point on the curve: the statistic
    is not noisy there, it is undefined. Until 2026-08-28 this filter was
    passed True for the probes and False for the fine-tuning arms, which
    published the probes on a well-defined subset and the arms on that subset
    plus their sign-flipped posts -- an asymmetry that depressed every arm's
    margin. It is now True on both paths and the parameter is kept only so the
    call sites stay explicit about it.

    `keys` names the columns one published row is identified by -- ("method",
    "variant") for the probes, ("arm",) for the fine-tuning cells. It is a list
    rather than a single column so that a method which splits its published
    rows is not averaged back together here.
    """
    keys = list(keys)
    d = frame[(frame["intervention"] == INTERVENTION)
              & (~frame["small_logit_gap"].astype(bool))]
    if positive_gap_only:
        d = d[d["clean_empty_logit_gap"] > 0]
    rows = []
    for name, g in d.groupby(keys, dropna=False):
        tgt = g[g["control"] == "target"].groupby("id")["normalized_aopc"].agg("max")
        ctl = g[g["control"] == CONTROL].groupby("id")["normalized_aopc"].agg("mean")
        common = tgt.index.intersection(ctl.index)
        diff = (tgt.loc[common] - ctl.loc[common]).to_numpy()
        margin, lo, hi = mean_ci(diff)
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row.update({"margin": margin, "margin_lo": lo, "margin_hi": hi,
                    "n_posts": int(len(common))})
        rows.append(row)
    return pd.DataFrame(rows)


def compute_probe_cells(model, wanted, seed, folds, allow):
    """Re-encode each probe's word deletions and score them. GPU, gated.

    Deletion changes the TEXT, so the activation cache cannot be reused and
    every variant goes back through the frozen backbone -- that is the only
    reason this half needs a model process at all.
    """
    from src import probe as probing
    from src.utils.common import (DENSE_PROBE_LAYER, load_config,
                                  require_model_process_approval,
                                  span_eval_ids, variants_of)
    from src.utils.frozen_backbone import FrozenBackbone, probe_word_deletion_rows

    cfg = load_config()
    require_model_process_approval(
        cfg, cli_override=allow, command="src.stage2.word_deletion --compute")
    from src.utils.common import probe_layer_of

    reduction = cfg["audit"]["word_score_aggregation"]
    max_len = int(cfg["full_finetune"]["max_len"])
    ctxs: dict = {}
    backbones: dict = {}
    loaded = None

    def at(layer_: int):
        """(ctx, backbone) at one cached layer; one set of weights, re-read
        at another layer for a final-layer probe."""
        nonlocal loaded
        if layer_ not in ctxs:
            ctxs[layer_] = probing.build_context(model, layer_, cfg, seed)
        if layer_ not in backbones:
            if loaded is None:
                loaded = FrozenBackbone.load(model, layer_, max_len)
                backbones[layer_] = loaded
            else:
                backbones[layer_] = FrozenBackbone(model, layer_, max_len,
                                                   loaded.model, loaded.tokenizer)
        return ctxs[layer_], backbones[layer_]

    rows = []
    for method in wanted:
        layer = probe_layer_of(method, model)
        ctx, backbone = at(layer)
        variants = list(variants_of(method))
        for variant in variants:
            tag = f"{method}[{variant}]" if variant else method
            for fold in folds:
                fitted = probing.FITTERS[method](probing.for_fold(ctx, fold))
                if fitted.token_scores is None or fitted.score_vectors is None:
                    print(f"  {tag}: {fitted.token_source} — skipped")
                    break
                eval_ids = span_eval_ids(ctx.posts, cfg, fold)
                sub = ctx.posts[ctx.posts["id"].isin(eval_ids)]
                print(f"  {tag} fold {fold}: {len(sub)} posts", flush=True)
                for _, row in sub.iterrows():
                    pid = row["id"]
                    ws = word_scores_from_tokens(fitted, ctx, pid, reduction)
                    n_words = max(ws) + 1 if ws else 0
                    dense = np.full(n_words, np.nan)
                    for w, v in ws.items():
                        dense[w] = v
                    rows += probe_word_deletion_rows(
                        backbone=backbone, scorer=fitted.score_vectors,
                        text=row["text"], word_scores=dense,
                        cached=ctx.acts[pid]["vecs"], config=cfg, seed=seed,
                        method=method, model=model, variant=variant,
                        fold=fold, pid=pid)
    if not rows:
        return None
    _merge_probe_rows(model, pd.DataFrame(rows))
    return artifact_path("delete", "probes", model)


def compute_sae_probe_cells(model, methods, seed, folds, allow) -> None:
    """The two SAE probes' deletion cells. GPU, gated.

    Same intervention and same output rows as `compute_probe_cells`; only the
    layer and the readout differ. The deleted text is re-encoded at
    SAE_PROBE_LAYER and pushed through the pinned dictionary before the probe's
    own readout sees it (`src.utils.sae_frozen_probe`), because these two arms
    are not functions of the residual stream the dense audit re-encodes into.
    The ranking is still each probe's OWN token decomposition.
    """
    import numpy as np

    from src import probe as probing
    from src.utils.common import (load_config, require_model_process_approval,
                                  span_eval_ids)
    from src.utils.data_prep import load_layer
    from src.utils.frozen_backbone import FrozenBackbone, probe_word_deletion_rows
    from src.utils.sae_features import aggregate_word_scores
    from src.utils.sae_frozen_probe import fit_readouts

    cfg = load_config()
    require_model_process_approval(
        cfg, cli_override=allow, command="src.stage2.word_deletion --compute")
    reduction = cfg["audit"]["word_score_aggregation"]
    max_len = int(cfg["full_finetune"]["max_len"])

    rows = []
    backbone = None
    for method in methods:
        readouts, posts, layer = fit_readouts(method, model, folds, seed, cfg,
                                              allow)
        acts = load_layer(model, "lonxplain", layer)
        # One backbone for both methods: they share SAE_PROBE_LAYER, and the
        # weights are the largest thing this command holds.
        backbone = backbone or FrozenBackbone.load(model, layer, max_len)
        by_id = posts.set_index("id")
        for fold in folds:
            readout = readouts[fold]
            fold_ids = span_eval_ids(posts, cfg, fold)
            print(f"  {method} fold {fold}: {len(fold_ids)} posts", flush=True)
            for pid in fold_ids:
                row = by_id.loc[pid]
                vecs = np.asarray(acts[pid]["vecs"], dtype=np.float64)
                targets = probing.word_targets(
                    np.asarray(acts[pid]["offsets"]), row["text"],
                    int(row["span_start"]), int(row["span_end"]))
                ws = aggregate_word_scores(
                    readout.token_vectors_scores(vecs), targets.alignment,
                    reduction=reduction)
                rows += probe_word_deletion_rows(
                    backbone=backbone, scorer=readout.score_vectors,
                    text=row["text"], word_scores=ws, cached=vecs,
                    config=cfg, seed=seed, method=method, model=model,
                    variant="", fold=fold, pid=pid)
    if not rows:
        return
    _merge_probe_rows(model, pd.DataFrame(rows))


def _merge_probe_rows(model, frame) -> None:
    """Merge new probe rows into delete_probes_{model}.parquet.

    Shared by both probe paths: every probe writes into ONE file, so a wholesale
    write would drop the methods the run was not asked to score.
    """
    out = artifact_path("delete", "probes", model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_parquet(out)
        keys = set(zip(frame["method"], frame["variant"], frame["fold"]))
        old = old[~old.apply(
            lambda r: (r["method"], r["variant"], r["fold"]) in keys, axis=1)]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_parquet(out, index=False)
    print(f"  wrote {out.name} ({len(frame)} rows)")


def probe_margins(model, wanted, compute, allow, *, seed=0, folds=None):
    raw = artifact_path("delete", "probes", model)
    if compute:
        # Recompute the methods this file does not already carry, not just the
        # whole file when it is absent: the probe cells live in ONE parquet, so
        # gating on the file's existence made `--compute --method mlp_probe`
        # silently do nothing as soon as any other probe had ever been scored.
        have = (set(pd.read_parquet(raw, columns=["method"])["method"].unique())
                if raw.exists() else set())
        todo = sorted(set(wanted) - have - set(CELLS_FROM))
        dense = [m for m in todo if m not in SAE_METHODS]
        sae = [m for m in todo if m in SAE_METHODS]
        if dense:
            compute_probe_cells(model, dense, seed, folds or [0, 2, 4, 6, 8],
                                allow)
        if sae:
            compute_sae_probe_cells(model, sae, seed,
                                    folds or [0, 2, 4, 6, 8], allow)
    if not raw.exists():
        print(f"  probes: {raw.name} missing — re-run with "
              "--compute --allow-model-process (deletion re-encodes the text, "
              "so it needs the backbone).")
        return pd.DataFrame()
    frame = pd.read_parquet(raw)
    if "variant" not in frame.columns:
        frame["variant"] = ""
    frame["variant"] = frame["variant"].fillna("")
    table = _margins(frame[frame["method"].isin(wanted)], ("method", "variant"),
                     positive_gap_only=True)
    table["group"] = "probe"
    return table


def compute_arm_cells(model, wanted, folds, seed, allow):
    """Score the arms' word deletions from their RETAINED checkpoints. GPU.

    This used to run inside src.stage1.train, while the freshly trained weights
    were still in memory. It lives here now: a metric is decided by its own CLI,
    not by whichever training run happened to go first. The cost is that a cell
    can only be scored while its checkpoint exists -- fold 0 always, the rest
    with `src.stage1.train --keep-checkpoint`.
    """
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.common import (checkpoint_path, load_config,
                                  require_model_process_approval, span_eval_ids)
    from src.utils.data_prep import load_corpus
    from src.utils.faithfulness import arm_word_deletion_rows
    from src.utils.sae_features import (PooledClassifierHookAdapter,
                                        aggregate_word_scores, align_post_tokens)

    cfg = load_config()
    posts, _ = load_corpus("lonxplain")
    ftcfg = cfg["full_finetune"]
    max_len = int(ftcfg["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]

    todo = []
    for method in wanted:
        if method in CELLS_FROM:
            continue        # another module writes these cells (CELLS_FROM)
        arm = ARM_OF[method]
        for cell in ([0] if method == "anchor" else folds):
            ck = checkpoint_path(arm, model, cell)
            if ck.exists():
                todo.append((method, arm, cell, ck))
            else:
                print(f"  {method} fold {cell}: no checkpoint — SKIPPED. "
                      f"Retain it with `src.stage1.train --method {method} "
                      f"--folds {cell} --keep-checkpoint`; this never trains.")
    if not todo:
        return
    # Gate only once weights are actually about to load, so a run that can
    # score nothing reports that instead of demanding an approval it won't use.
    require_model_process_approval(
        cfg, cli_override=allow, command="src.stage2.word_deletion --compute")

    for method, arm, cell, ck in todo:
        model_obj, tok = ft.load_checkpoint(ck, model)
        try:
            adapter = PooledClassifierHookAdapter(model_obj)
            device = next(model_obj.parameters()).device
            eval_ids = span_eval_ids(posts, cfg, cell)
            sub = posts[posts["id"].isin(eval_ids)]
            print(f"  {method} fold {cell}: {len(sub)} posts", flush=True)
            rows = []
            for _, row in sub.iterrows():
                alignment = align_post_tokens(
                    tok, row["text"], max_len=max_len,
                    span_start=int(row["span_start"]),
                    span_end=int(row["span_end"]))
                ids, attn, pm = ft.build_inputs(tok, [row["text"]], max_len)
                with torch.no_grad():
                    scores = ft.token_scores(model_obj, ids.to(device),
                                             attn.to(device), pm.to(device))
                ws = aggregate_word_scores(scores, alignment, reduction=reduction)
                for r in arm_word_deletion_rows(
                        adapter, tok, row["text"], ws, config=cfg,
                        finetune_config=ftcfg, seed=seed):
                    rows.append({"arm": arm, "model": model, "fold": cell,
                                 "id": row["id"], **r})
            out = artifact_path("delete", arm, model, fold=cell)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"    wrote {out.name} ({len(rows)} rows)")
        finally:
            del model_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def arm_margins(model, wanted):
    frames = []
    for method in wanted:
        arm = ARM_OF[method]
        files = sorted(globlib.glob(str(
            artifact_path("delete", arm, model, fold="*"))))
        if not files:
            print(f"  {method}: no audit cells on disk — train the arm first "
                  f"(src.stage1.train --method {method})")
            continue
        frames.append(pd.concat([pd.read_parquet(f) for f in files],
                                ignore_index=True))
    if not frames:
        return pd.DataFrame()
    table = _margins(pd.concat(frames, ignore_index=True), ("arm",),
                     positive_gap_only=True)
    table["method"] = table["arm"].map(METHOD_OF_ARM)
    table["variant"] = ""
    table["group"] = "baseline"
    return table.drop(columns=["arm"])


def cell_margins(model, wanted):
    """Rows for methods whose delete_* cells another module writes under the
    method's own name (CELLS_FROM entries that are neither arms nor probes:
    the two references' own decomposition)."""
    frames = []
    for method in sorted(wanted):
        files = sorted(globlib.glob(str(
            artifact_path("delete", method, model, fold="*"))))
        if not files:
            print(f"  {method}: no cells on disk — run {CELLS_FROM[method]}")
            continue
        f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        f["arm"] = method
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    table = _margins(pd.concat(frames, ignore_index=True), ("arm",),
                     positive_gap_only=True)
    table["method"] = table["arm"]
    table["variant"] = ""
    table["group"] = "baseline"
    return table.drop(columns=["arm"])


def metric_main():
    ap = argparse.ArgumentParser(
        description=f"{INTERVENTION} margins -> results/word_deletion.csv")
    add_method_args(ap, choices=METHODS)
    ap.add_argument("--compute", action="store_true",
                    help="probes: re-encode and score the missing raw cells "
                         "through the frozen backbone (GPU)")
    ap.add_argument("--allow-model-process", action="store_true")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    add_device_args(ap)
    args = ap.parse_args()

    methods = selected_methods(args, choices=METHODS)
    model = resolve_model(args.model)
    # Aggregating cells off disk is CPU work; --compute re-encodes deleted
    # text through the backbone and the arms, and needs a card of its own.
    pin_visible_devices(args.devices, need_gpu=bool(args.compute),
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)

    print(f"{INTERVENTION} margins — model {model}")
    parts = []
    probe_wanted = {m for m in methods if group_of(m) == "probe"}
    arm_wanted = {m for m in methods if m in ARM_OF}
    if probe_wanted:
        p = probe_margins(model, probe_wanted, args.compute,
                          args.allow_model_process, seed=args.seed,
                          folds=args.folds)
        if not p.empty:
            parts.append(p)
    if arm_wanted:
        if args.compute:
            compute_arm_cells(model, arm_wanted, args.folds, args.seed,
                              args.allow_model_process)
        a = arm_margins(model, arm_wanted)
        if not a.empty:
            parts.append(a)
    cell_wanted = {m for m in methods if m in CELLS_FROM and m not in ARM_OF
                   and group_of(m) != "probe"}
    if cell_wanted:
        c = cell_margins(model, cell_wanted)
        if not c.empty:
            parts.append(c)
    if not parts:
        raise SystemExit("no rows — see the messages above")

    table = pd.concat(parts, ignore_index=True)
    table["model"] = model
    table["intervention"] = INTERVENTION
    table["control"] = CONTROL
    table["variant"] = table.get("variant", "").fillna("") \
        if "variant" in table else ""
    table = table[["method", "group", "model", "variant", "margin", "margin_lo",
                   "margin_hi", "n_posts", "intervention", "control"]]
    for _, r in table.iterrows():
        tag = f"{r['method']}[{r['variant']}]" if r["variant"] else r["method"]
        print(f"  {tag:<28} margin {r['margin']:+.3f} "
              f"[{r['margin_lo']:+.3f}, {r['margin_hi']:+.3f}] "
              f"({r['n_posts']} posts)")

    if OUT_CSV.exists():
        # `variant` is new: rows written before it existed read back as NaN and
        # must compare equal to the empty variant they were, or a re-run would
        # append a duplicate beside every one of them instead of replacing it.
        old = pd.read_csv(OUT_CSV)
        if "variant" not in old.columns:
            old["variant"] = ""
        old["variant"] = old["variant"].fillna("")
        keys = set(zip(table["method"], table["model"], table["variant"]))
        old = old[~old.apply(
            lambda r: (r["method"], r["model"], r["variant"]) in keys, axis=1)]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["model", "margin"], ascending=[True, False])
    table.to_csv(OUT_CSV, index=False)
    print(f"\nwrote results/word_deletion.csv ({len(table)} rows)")


if __name__ == "__main__":
    metric_main()
