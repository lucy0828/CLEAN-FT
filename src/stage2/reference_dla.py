"""Own decomposition (DLA) of the two reference detectors -> Stage 2 cells.

How to run (GPU for MentalBERT; GloVe runs anywhere):
    uv run python -m src.stage2.reference_dla --method mentalbert glove_bigru \\
        --folds 0 2 4 6 8 --allow-model-process
    uv run python -m src.stage2.dla_auprc --method mentalbert glove_bigru      # CPU
    uv run python -m src.stage2.word_deletion --method mentalbert glove_bigru  # CPU

Both references pool by the MEAN over their positions and read the pooled
vector with a linear head, so their score is exactly the sum the study's
DLA decomposes:

    f(x) = sum_t (1/n) (w . X_t) + b

with X_t the encoder's last hidden state at token t (MentalBERT) or the
BiGRU's hidden state at word t (GloVe+BiGRU). Each position's share is the
same quantity Eq. decomp assigns under mean pooling (alpha_t = 1/n); the
decomposition is exact, and `decomposition_max_abs_error` records the check.
MentalBERT's WordPiece shares are aggregated to words with the pinned
reduction (max); GloVe's positions ARE the words (`glove_bigru.encode_words`
walks `words(text)` in order, OOV words as zero vectors).

The cells land under the method's own name (`plaus_mentalbert_*`,
`delete_glove_bigru_*`), filed under the study default backbone label
exactly as their occlusion cells are (src/utils/common.py, CELLS_FROM), and
the two aggregators read them back as-is. Word deletion runs on text through
`src.stage2.blackbox.text_word_deletion_rows`, the same audit the occlusion
cells use, with `ranking_source="dla"`.

These are the model's OWN scores, so the cells fill the DLA AUPRC and the
DLA ΔAOPC columns of the reference rows. Neither model has an attention-pooled
variant: both were trained under one fixed recipe each.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.utils.common import (DEFAULT_MODEL, RESULTS, add_device_args, artifact_path,
                              load_config, pin_visible_devices,
                              require_model_process_approval, span_eval_ids)

METHODS = ("mentalbert", "glove_bigru")
RANKING_SOURCE = "dla"


# --------------------------------------------------------------- the terms


def mentalbert_word_terms(box, text: str, *, reduction: str):
    """(per-word DLA over words(text), clean logit, |sum + b - f(x)|,
    n_tokens, truncated) for the MentalBERT classifier."""
    import torch

    from src.probe import word_targets
    from src.utils.data_prep import words
    from src.utils.sae_features import aggregate_word_scores

    tok, model, max_len = box.tok, box.model, box.max_len
    full = tok(text, truncation=False)["input_ids"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].numpy()
    ids, attn = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    with torch.no_grad():
        h = model.backbone(input_ids=ids, attention_mask=attn).last_hidden_state[0].float()
        w = model.head.weight[0].float()
        b = model.head.bias.float()
        m = attn[0].float()
        terms = (h @ w) * m / m.sum().clamp(min=1e-6)
        logit = float(model(ids, attn)[0])
        err = abs(float(terms.sum() + b[0]) - logit)
    terms = terms.cpu().numpy().astype(np.float64)
    n_words = len(words(text))
    if not len(offsets):
        return np.full(n_words, np.nan), logit, err, 0, False
    # special tokens carry (0, 0) offsets and overlap no word
    targets = word_targets(offsets, text, -1, -1)
    ws = aggregate_word_scores(terms, targets.alignment, reduction=reduction)
    return ws, logit, err, int(len(offsets)), bool(len(full) > len(offsets))


def glove_word_terms(box, text: str, *, max_len: int):
    """(per-word DLA over words(text), clean logit, |sum + b - f(x)|,
    n_words_in, truncated) for the GloVe+BiGRU classifier."""
    import torch

    from src.baselines import glove_bigru as gb
    from src.utils.data_prep import words

    model = box.model
    post_words = words(text)
    n_words = len(post_words)
    ws = np.full(n_words, np.nan)
    if n_words == 0:
        logit = float(box.score_texts([text])[0])
        return ws, logit, 0.0, 0, False
    dev = next(model.parameters()).device
    x, mask = gb._batch([text], max_len)
    x, mask = x.to(dev), mask.to(dev)
    with torch.no_grad():
        h, _ = model.gru(x)
        w = model.head.weight[0]
        b = model.head.bias
        n = mask[0].sum().clamp(min=1.0)
        terms = (h[0] @ w) * mask[0] / n
        logit = float(model(x, mask)[0])
        err = abs(float(terms.sum() + b[0]) - logit)
    n_in = int(mask[0].sum().item())
    ws[:n_in] = terms[:n_in].cpu().numpy().astype(np.float64)
    return ws, logit, err, n_in, bool(n_words > n_in)


def word_terms(box, method: str, text: str, cfg: dict):
    """Dispatch on the reference: (word scores, logit, error, n, truncated)."""
    reduction = cfg["audit"]["word_score_aggregation"]
    if method == "mentalbert":
        return mentalbert_word_terms(box, text, reduction=reduction)
    if method == "glove_bigru":
        return glove_word_terms(box, text, max_len=int(cfg["glove_bigru"]["max_len"]))
    raise ValueError(method)


def open_box(method: str, cell: int, cfg: dict, batch_size: int):
    from src.stage2.blackbox import GloVeBox, MentalBERTBox

    if method == "mentalbert":
        return MentalBERTBox(method, DEFAULT_MODEL, cell, cfg, batch_size)
    return GloVeBox(method, DEFAULT_MODEL, cell, cfg, batch_size)


# ---------------------------------------------------------------- one fold


def run_cell(method: str, cell: int, cfg: dict, posts, *, batch_size: int,
             seed: int, limit: int | None, smoke: bool) -> None:
    from src.stage2.blackbox import text_word_deletion_rows, word_auprc_from_scores

    model_name = DEFAULT_MODEL
    eval_ids = span_eval_ids(posts, cfg, cell)
    sub = posts[posts["id"].isin(eval_ids)]
    if limit:
        sub = sub.head(limit)
    box = open_box(method, cell, cfg, batch_size)
    if not box.ok:
        print(f"  {method} fold {cell}: no checkpoint -- SKIPPED", flush=True)
        return
    print(f"  {method} fold {cell}: {len(sub)} posts, {box.describe}", flush=True)
    plaus_rows, delete_rows = [], []
    try:
        for n_done, (_, row) in enumerate(sub.iterrows(), start=1):
            text, pid = str(row["text"]), row["id"]
            span = (int(row["span_start"]), int(row["span_end"]))
            ws, clean, err, n_tok, truncated = word_terms(box, method, text, cfg)
            plaus_rows.append({
                "arm": method, "model": model_name, "fold": cell, "id": pid,
                "token_auprc": float("nan"), "n_content_tokens": n_tok,
                "n_gold_tokens": 0, "gold_base_rate": float("nan"),
                **word_auprc_from_scores(text, ws, span),
                "clean_logit": clean, "decomposition_max_abs_error": err,
                "truncated": truncated, "ranking_source": RANKING_SOURCE})
            if np.isfinite(ws).any():
                for r in text_word_deletion_rows(box.score_texts, text, ws, config=cfg,
                                                 seed=seed, ranking_source=RANKING_SOURCE):
                    delete_rows.append({"arm": method, "model": model_name,
                                        "fold": cell, "id": pid, **r})
            if n_done == 1 or n_done % 20 == 0:
                print(f"    {n_done}/{len(sub)}  DLA word AUPRC so far "
                      f"{np.nanmean([p['word_auprc'] for p in plaus_rows]):.3f}, "
                      f"max decomposition error {max(p['decomposition_max_abs_error'] for p in plaus_rows):.2e}",
                      flush=True)
    finally:
        box.release()

    for stage, rows in (("plaus", plaus_rows), ("delete", delete_rows)):
        out = artifact_path(stage, method, model_name, fold=cell, smoke=smoke)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(out)
        print(f"    wrote {out.name} ({len(rows)} rows)", flush=True)
    w = np.asarray([p["word_auprc"] for p in plaus_rows], dtype=float)
    print(f"  {method} fold {cell}: DLA word AUPRC {np.nanmean(w):.4f} over {len(w)} posts",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-model-process", action="store_true")
    add_device_args(ap)
    args = ap.parse_args()
    if args.limit and not args.smoke:
        raise SystemExit("--limit writes a partial cell; add --smoke")
    pin_visible_devices(args.devices, need_gpu=("mentalbert" in args.method),
                        min_free_gb=8, wait_minutes=args.wait_for_gpu)
    cfg = load_config()
    require_model_process_approval(cfg, cli_override=args.allow_model_process,
                                   command="src.stage2.reference_dla")
    from src.utils.data_prep import load_corpus

    posts, _ = load_corpus("lonxplain")
    print(f"reference DLA -- methods {args.method}, folds {args.folds}")
    for method in args.method:
        for cell in args.folds:
            run_cell(method, cell, cfg, posts, batch_size=args.batch_size,
                     seed=args.seed, limit=args.limit, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
