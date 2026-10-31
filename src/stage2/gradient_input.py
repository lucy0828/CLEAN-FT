"""Gradient x input through the backbone -> Stage 2 cells for three methods.

How to run (GPU; one fold per card is the natural unit):
    uv run python -m src.stage2.gradient_input --method gxi_ft --model gemma2 --folds 0 --allow-model-process
    uv run python -m src.stage2.gradient_input --method last_ft last_token_probe --model llama --folds 0 2 4 6 8 --allow-model-process
    uv run python -m src.stage2.gradient_input --method gxi_ft --model gemma2 --folds 0 --limit 3 --smoke

then aggregate exactly as for any arm:
    uv run python -m src.stage2.dla_auprc      --method gxi_ft last_ft last_token_probe --model gemma2
    uv run python -m src.stage2.word_deletion  --method gxi_ft last_ft last_token_probe --model gemma2

THE THREE METHODS
-----------------
  gxi_ft            gradient x input over the FULL FT checkpoint: the same
                    classifier as `full_ft`, so it shares full_ft's detection
                    row (EXPLAINER_BASE) and differs from it in the word
                    ranking alone: the gradient-based post-hoc explainer
                    control of PAPER.md section E.
  last_ft           the last-token read-out fine-tune (fine_tune.ARMS
                    `finetune_last_random`). Its own decomposition is a
                    one-hot on the final post token, which is not a ranking,
                    so gradient x input IS its word attribution: the credit
                    the last position absorbed through causal attention is
                    handed back to the input positions it came from.
  last_token_probe  the frozen last-token linear probe at DENSE_PROBE_LAYER
                    (src.probe.fit_last_token_probe), for the same reason.
                    The backbone is run with gradients enabled and the
                    probe's affine read-out is applied to the last post
                    token's state at the probe layer.

THE ATTRIBUTION
---------------
For a post with input embeddings E [T, H] and decision logit f(E),

    s_t = sum_h  E_th * d f / d E_th          (gradient x input, per token)

read only on the post tokens. It is the first-order Taylor share of each
token in the logit -- a position's contribution INCLUDING what flows to the
read-out through attention -- and unlike the pooled decomposition it is
defined for any read-out. It is also the one-step end of the Integrated
Gradients path, so it is the cheapest member of the family the method text
already uses for the MLP probes (there in the pooled feature space; here
through the whole backbone). Sub-word scores are reduced to words with the
pinned aggregation, and the deletion audit ranks by the resulting word score
exactly as it does for every other arm. `decomposition_max_abs_error` is
NaN: a first-order share has no completeness identity to check.

The cells are written under each method's own name (its arm name where it
has one), so `src.stage2.dla_auprc` and `src.stage2.word_deletion` read them
through CELLS_FROM without special cases; the probe's deletion rows join the
shared delete_probes_{model}.parquet like every other probe's.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.utils.common import (ARM_OF, CELLS_FROM, DENSE_PROBE_LAYER,
                              EXPLAINER_BASE, add_device_args, artifact_path,
                              checkpoint_path, load_config, min_free_gb_for,
                              pin_visible_devices,
                              require_model_process_approval, resolve_model,
                              span_eval_ids)

METHODS = ("gxi_ft", "last_ft", "last_token_probe")
RANKING_SOURCE = "gradient_x_input"


# ------------------------------------------------------------ the attribution


def gradient_x_input(logit_fn, embed_fn, input_ids, attention_mask, post_mask):
    """[T] gradient x input on the post tokens of ONE sequence.

    `embed_fn(ids) -> [1, T, H]` embeddings, `logit_fn(inputs_embeds,
    attention_mask, post_mask) -> [1]` the decision. Everything off the post
    mask is zero.
    """
    import torch

    embeds = embed_fn(input_ids).detach().float().requires_grad_(True)
    logit = logit_fn(embeds, attention_mask, post_mask)
    (grad,) = torch.autograd.grad(logit.sum(), embeds)
    scores = (grad * embeds).sum(-1)[0]                      # [T]
    scores = scores * post_mask[0].to(scores.dtype)
    return scores.detach().cpu().numpy().astype(np.float64), float(logit.item())


# ------------------------------------------------------ the two read-outs


def classifier_scorers(model):
    """(logit_fn, embed_fn) for a fine_tune.PooledClassifier checkpoint."""
    import torch

    backbone = model.backbone
    embedding = backbone.get_input_embeddings()
    # Only the input embeddings need a gradient; freezing the parameters keeps
    # autograd from tracking every weight of an 8B model for one vector.
    for param in model.parameters():
        param.requires_grad_(False)

    def embed_fn(ids):
        e = embedding(ids)
        # Gemma scales embeddings by sqrt(H) inside the model, after the
        # lookup; that scaling is part of the forward we differentiate
        # through, so the raw table output is the right leaf here.
        return e.to(torch.float32)

    def logit_fn(embeds, attention_mask, post_mask):
        return model(inputs_embeds=embeds.to(next(backbone.parameters()).dtype),
                     attention_mask=attention_mask, post_mask=post_mask)

    return logit_fn, embed_fn


class FrozenLastTokenReadout:
    """The last-token probe as a differentiable function of the input.

    Runs the frozen causal LM with gradients enabled, takes hidden_states[k]
    (k = DENSE_PROBE_LAYER, the activation cache's index convention) at the
    LAST post token, and applies the probe's affine read-out a . h + b -- the
    same numbers `src.probe.fit_last_token_probe` scores posts with.
    """

    def __init__(self, backbone, layer: int, a: np.ndarray, b: float):
        import torch

        self.backbone = backbone
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.layer = layer
        self.a = torch.tensor(a, dtype=torch.float32,
                              device=next(backbone.parameters()).device)
        self.b = float(b)
        self.embedding = backbone.get_input_embeddings()

    def embed_fn(self, ids):
        import torch

        return self.embedding(ids).to(torch.float32)

    def logit_fn(self, embeds, attention_mask, post_mask):
        import torch

        dtype = next(self.backbone.parameters()).dtype
        out = self.backbone(inputs_embeds=embeds.to(dtype),
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        h = out.hidden_states[self.layer].float()                # [1, T, H]
        m = post_mask.to(torch.float32)
        last = ((m.cumsum(1) == m.sum(1, keepdim=True)) & (m > 0)).float()
        pooled = (h * last.unsqueeze(-1)).sum(1)                  # [1, H]
        return pooled @ self.a + self.b


# ------------------------------------------------------------- one fold


def _probe_affine(model_name: str, cell: int, cfg: dict, seed: int):
    """(a, b) of the last-token probe refitted out of fold, as the study fits it."""
    from src import probe as probing

    layer = DENSE_PROBE_LAYER[model_name]
    ctx = probing.build_context(model_name, layer, cfg, seed)
    fold_ctx = probing.for_fold(ctx, cell)
    fitted = probing.FITTERS["last_token_probe"](fold_ctx)
    _, a, b = probing.last_token_affine(fold_ctx)
    return np.asarray(a, dtype=np.float64), float(b), fitted, ctx


def run_cell(method: str, model_name: str, cell: int, cfg: dict, posts, *,
             seed: int, limit: int | None, smoke: bool,
             dtype: str = "bfloat16") -> None:
    import torch

    from src.baselines import fine_tune as ft
    from src.utils.dla import _word_plausibility
    from src.utils.faithfulness import arm_word_deletion_rows
    from src.utils.sae_features import (PooledClassifierHookAdapter,
                                        aggregate_word_scores,
                                        align_post_tokens,
                                        token_plausibility_auprc)

    ftcfg = cfg["full_finetune"]
    max_len = int(ftcfg["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]
    name = ARM_OF.get(method, method)

    eval_ids = span_eval_ids(posts, cfg, cell)
    sub = posts[posts["id"].isin(eval_ids)]
    if limit:
        sub = sub.head(limit)

    # --- the read-out under test
    is_probe = method == "last_token_probe"
    if is_probe:
        from src.utils.frozen_backbone import FrozenBackbone
        from src.utils.frozen_backbone import probe_word_deletion_rows

        a, b, fitted, ctx = _probe_affine(model_name, cell, cfg, seed)
        layer = DENSE_PROBE_LAYER[model_name]
        frozen = FrozenBackbone.load(model_name, layer, max_len)
        backbone, tok = frozen.model, frozen.tokenizer
        if dtype == "float32":
            backbone.float()
        readout = FrozenLastTokenReadout(backbone.model if hasattr(backbone, "model")
                                         else backbone, layer, a, b)
        logit_fn, embed_fn = readout.logit_fn, readout.embed_fn
        adapter = None
        print(f"  {method} fold {cell}: {len(sub)} posts, frozen {model_name} "
              f"layer {layer}", flush=True)
    else:
        base = EXPLAINER_BASE.get(method, method)
        checkpoint = checkpoint_path(ARM_OF[base], model_name, cell)
        if not checkpoint.exists():
            print(f"  {method} fold {cell}: no {base} checkpoint at "
                  f"{checkpoint} — SKIPPED (train it with --keep-checkpoint; "
                  f"this command never trains).", flush=True)
            return
        model, tok = ft.load_checkpoint(checkpoint, model_name)
        if dtype == "float32":
            # The gradient of a 26-42 block stack in bf16 is a noisy object;
            # the attribution (not the decision) is taken in float32 when asked.
            model.float()
        logit_fn, embed_fn = classifier_scorers(model)
        adapter = PooledClassifierHookAdapter(model)
        print(f"  {method} fold {cell}: {len(sub)} posts from {checkpoint.name}",
              flush=True)

    device = next((adapter.classifier if adapter else backbone).parameters()).device
    plaus_rows, delete_rows = [], []
    try:
        for n_done, (_, row) in enumerate(sub.iterrows(), start=1):
            text = str(row["text"])
            span = (int(row["span_start"]), int(row["span_end"]))
            ids, attn, pm = ft.build_inputs(tok, [text], max_len)
            ids, attn, pm = ids.to(device), attn.to(device), pm.to(device)
            seq, clean = gradient_x_input(logit_fn, embed_fn, ids, attn, pm)

            alignment = align_post_tokens(tok, text, max_len=max_len,
                                          span_start=span[0], span_end=span[1])
            n_content = int(alignment.token_indices.size)
            n_gold = int(alignment.gold_mask.sum())
            plaus_rows.append({
                "arm": name, "model": model_name, "fold": cell, "id": row["id"],
                "token_auprc": token_plausibility_auprc(seq, alignment),
                "n_content_tokens": n_content,
                "n_gold_tokens": n_gold,
                "gold_base_rate": (n_gold / n_content) if n_content else float("nan"),
                **_word_plausibility(seq, alignment, text, span[0], span[1],
                                     reduction),
                "clean_logit": clean,
                "decomposition_max_abs_error": float("nan"),
                "truncated": bool(alignment.truncated),
                "ranking_source": RANKING_SOURCE,
            })
            word_scores = aggregate_word_scores(seq, alignment, reduction=reduction)
            if np.isfinite(word_scores).any():
                if is_probe:
                    delete_rows += probe_word_deletion_rows(
                        backbone=frozen, scorer=fitted.score_vectors, text=text,
                        word_scores=word_scores,
                        cached=ctx.acts[row["id"]]["vecs"], config=cfg,
                        seed=seed, method=method, model=model_name,
                        variant="", fold=cell, pid=row["id"],
                        ranking_source=RANKING_SOURCE)
                else:
                    for r in arm_word_deletion_rows(
                            adapter, tok, text, word_scores, config=cfg,
                            finetune_config=ftcfg, seed=seed,
                            ranking_source=RANKING_SOURCE):
                        delete_rows.append({"arm": name, "model": model_name,
                                            "fold": cell, "id": row["id"], **r})
            if n_done == 1 or n_done % 20 == 0:
                print(f"    {n_done}/{len(sub)}  word AUPRC so far "
                      f"{np.nanmean([p['word_auprc'] for p in plaus_rows]):.3f}",
                      flush=True)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = artifact_path("plaus", name, model_name, fold=cell, smoke=smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pd.DataFrame(plaus_rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    print(f"    wrote {out.name} ({len(plaus_rows)} rows)", flush=True)
    if is_probe:
        if smoke:
            out = artifact_path("delete", "probes", model_name, smoke=True)
            pd.DataFrame(delete_rows).to_parquet(out, index=False)
            print(f"    wrote {out} ({len(delete_rows)} rows)", flush=True)
        else:
            from src.stage2.word_deletion import _merge_probe_rows
            _merge_probe_rows(model_name, pd.DataFrame(delete_rows))
    else:
        out = artifact_path("delete", name, model_name, fold=cell, smoke=smoke)
        tmp = out.with_suffix(".parquet.tmp")
        pd.DataFrame(delete_rows).to_parquet(tmp, index=False)
        tmp.replace(out)
        print(f"    wrote {out.name} ({len(delete_rows)} rows)", flush=True)
    w = np.asarray([p["word_auprc"] for p in plaus_rows], dtype=float)
    print(f"  {method} fold {cell}: word AUPRC {np.nanmean(w):.4f} over "
          f"{int(np.isfinite(w).sum())}/{len(w)} scorable posts", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="gradient x input through the backbone -> plaus_* and "
                    "delete_* cells for gxi_ft, last_ft, last_token_probe")
    ap.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N span-eval posts of a fold")
    ap.add_argument("--smoke", action="store_true",
                    help="write under results/stage2/smoke/ (never a study cell)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float32"),
                    help="backbone precision for the attribution pass")
    ap.add_argument("--allow-model-process", action="store_true")
    add_device_args(ap)
    args = ap.parse_args()
    if args.limit and not args.smoke:
        raise SystemExit("--limit writes a partial cell; add --smoke")
    for m in args.method:
        assert CELLS_FROM[m] == "src.stage2.gradient_input", m

    model = resolve_model(args.model)
    pin_visible_devices(args.devices, need_gpu=True,
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)
    cfg = load_config()
    require_model_process_approval(
        cfg, cli_override=args.allow_model_process,
        command="src.stage2.gradient_input")

    from src.utils.data_prep import load_corpus
    posts, _ = load_corpus("lonxplain")
    print(f"gradient x input — model {model}, methods {args.method}, "
          f"folds {args.folds}")
    for method in args.method:
        for cell in args.folds:
            run_cell(method, model, cell, cfg, posts, seed=args.seed,
                     limit=args.limit, smoke=args.smoke, dtype=args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
