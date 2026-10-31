"""Occlusion (leave-one-word-out) on the principal read-outs, beside their own
decomposition -> occl_* Stage 2 cells, per-word agreement rows, and ERASER
sufficiency / comprehensiveness in probability space.

How to run (GPU; one fold per card is the natural unit):
    uv run python -m src.stage2.occlusion --method occl_attn_ft --model gemma2 --folds 0 --allow-model-process
    uv run python -m src.stage2.occlusion --method occl_attention_probe occl_midlayer_attn_lp_ft --model llama --allow-model-process
    uv run python -m src.stage2.occlusion --collect              # CPU: the two summary tables

then the explainer cells aggregate exactly as for any method:
    uv run python -m src.stage2.dla_auprc     --method occl_attn_ft ... --model gemma2
    uv run python -m src.stage2.word_deletion --method occl_attn_ft ... --model gemma2

WHAT THIS ANSWERS
-----------------
The reviewer's item: one UNIFORM input-level attribution on the four principal
models -- Attn-FT, Attn-ML-FT, Attn-Linear, Attn-LP-ML-FT -- plus the rank
correlation between it and each model's own decomposition (DLA). Occlusion
is the attribution: a word's score is f(x) - f(x \\ word), the logit change
when that one word is deleted from the text (`src.utils.faithfulness.
delete_words`, the study's one deletion primitive), read on the same
classifier through a black-box text scorer (ArmBox / ProbeBox in
src.stage2.blackbox). It is model-agnostic, word-native, deterministic,
and costs one forward per word.

Per post it writes:
  * the occlusion ranking's own plausibility and faithfulness cells
    (`plaus_occl_*`, `delete_occl_*`), under the method name `occl_<base>`,
    so the aggregators publish them beside every other method;
  * an `attrib_*` row per WORD with both scores (own decomposition and
    occlusion), and per post their Spearman rank correlation over the words
    both scored, plus the Jaccard overlap of their top-5 words;
  * an `eraser_*` row per k of the faithfulness grid for the model's OWN
    ranking: comprehensiveness p(x) - p(x \\ top-k) and sufficiency
    p(x) - p(top-k only) in PROBABILITY space (DeYoung et al. 2020), with
    top-k sized exactly as the deletion audit sizes it (a percentage of the
    positively-scored words, at least one). `--collect` averages both over
    the grid per method into results/eraser.csv and the agreement into
    results/attribution_agreement.csv.

Agreement is the finding the reviewer asks for: if DLA and occlusion rank the
words alike on the probe and unlike on the fine-tune, the decomposition is
not the thing that moved -- the model's dependence on its words is.
"""

from __future__ import annotations

import argparse
import glob as globlib
import sys

import numpy as np
import pandas as pd

from src.utils.common import (ARM_OF, CELLS_FROM, DENSE_PROBE_LAYER,
                              EXPLAINER_BASE, RESULTS, add_device_args,
                              artifact_path, group_of, load_config, mean_ci,
                              min_free_gb_for, pin_visible_devices,
                              require_model_process_approval, resolve_model,
                              span_eval_ids)

METHODS = ("occl_attn_ft", "occl_midlayer_attn_ft", "occl_midlayer_attn_lp_ft",
           "occl_attention_probe", "occl_attn_lp_ft",
           "occl_attn_lora_ft", "occl_attn_span_ft",
           "occl_final_attention_probe", "occl_last_token_probe",
           "occl_attention_mlp_probe", "occl_sae_attention_probe",
           "occl_sae_probe", "occl_mentalbert", "occl_glove_bigru")
SAE_BASES = ("sae_probe", "sae_attention_probe")
TEXT_BASES = ("mentalbert", "glove_bigru")   # raw-text detectors, no cache
RANKING_SOURCE = "occlusion"
TOP = 5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def k_word_count(n_words: int, k_percent: float) -> int:
    return max(1, min(n_words, int(np.ceil(n_words * k_percent / 100.0))))


# ------------------------------------------------------------ own rankings


def own_word_scores(box, base: str, text: str, pid: str, *, max_len: int,
                    reduction: str) -> np.ndarray:
    """The base method's OWN word scores (its decomposition, aggregated to
    words with the pinned reduction) -- what dla_auprc ranks by."""
    from src.utils.sae_features import aggregate_word_scores, align_post_tokens

    if hasattr(box, "own_word_scores"):          # the SAE read-outs
        row = box.posts.set_index("id").loc[pid]
        return box.own_word_scores(text, pid,
                                   (int(row["span_start"]), int(row["span_end"])),
                                   reduction)
    if group_of(base) == "probe":
        from src.probe import word_targets

        if box.fitted.token_scores is None:      # e.g. the last-token probe
            from src.utils.data_prep import words

            return np.full(len(words(text)), np.nan)
        row = box.ctx.posts.set_index("id").loc[pid]
        targets = word_targets(np.asarray(box.ctx.acts[pid]["offsets"]), text,
                               int(row["span_start"]), int(row["span_end"]))
        tok = np.asarray(box.fitted.token_scores(box.ctx.index_of[pid]),
                         dtype=np.float64)
        return aggregate_word_scores(tok, targets.alignment, reduction=reduction)
    import torch

    from src.baselines import fine_tune as ft

    device = next(box.model.parameters()).device
    alignment = align_post_tokens(box.tok, text, max_len=max_len)
    ids, attn, pm = ft.build_inputs(box.tok, [text], max_len)
    with torch.no_grad():
        tok = ft.token_scores(box.model, ids.to(device), attn.to(device),
                              pm.to(device))
    return aggregate_word_scores(tok, alignment, reduction=reduction)


def occlusion_scores(box, text: str) -> tuple[np.ndarray, float]:
    """(one score per `words(text)` entry, clean logit): f(x) - f(x \\ word)."""
    from src.utils.data_prep import words
    from src.utils.faithfulness import delete_words

    post_words = words(text)
    valid = [i for i, w in enumerate(post_words) if w["norm"]]
    scores = np.full(len(post_words), np.nan, dtype=float)
    variants = [text] + [delete_words(text, post_words, {i}) for i in valid]
    logits = np.asarray(box.score_texts(variants), dtype=float)
    clean = float(logits[0])
    for j, i in enumerate(valid, start=1):
        scores[i] = clean - float(logits[j])
    return scores, clean


def eraser_rows(box, text: str, own: np.ndarray, clean: float, k_grid) -> list[dict]:
    """Comprehensiveness and sufficiency in probability space for the OWN
    ranking, top-k sized as the deletion audit sizes it."""
    from src.utils.data_prep import words
    from src.utils.faithfulness import delete_words

    post_words = words(text)
    valid = np.flatnonzero(np.isfinite(own))
    active = valid[own[valid] > 0]
    ranked = active[np.argsort(-own[active], kind="stable")]
    if not len(ranked):
        return []
    variants, plans = [], []
    for k in k_grid:
        n_sel = min(k_word_count(len(ranked), k), len(ranked))
        top = set(ranked[:n_sel].tolist())
        variants.append(delete_words(text, post_words, top))                 # x \ top-k
        variants.append(delete_words(text, post_words, set(valid) - top))    # top-k only
        plans.append((k, n_sel))
    logits = np.asarray(box.score_texts(variants), dtype=float)
    p_clean = float(sigmoid(clean))
    rows = []
    for j, (k, n_sel) in enumerate(plans):
        p_comp, p_suff = float(sigmoid(logits[2 * j])), float(sigmoid(logits[2 * j + 1]))
        rows.append({"k_percent": k, "n_selected": n_sel, "n_active": int(len(ranked)),
                     "p_clean": p_clean, "p_without_topk": p_comp,
                     "p_topk_only": p_suff,
                     "comprehensiveness": p_clean - p_comp,
                     "sufficiency": p_clean - p_suff})
    return rows


def agreement(own: np.ndarray, occl: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    both = np.flatnonzero(np.isfinite(own) & np.isfinite(occl))
    out = {"n_words_both": int(len(both)), "spearman": float("nan"),
           "top5_jaccard": float("nan")}
    if len(both) >= 3:
        rho = spearmanr(own[both], occl[both]).correlation
        out["spearman"] = float(rho) if np.isfinite(rho) else float("nan")
        n = min(TOP, len(both))
        a = set(both[np.argsort(-own[both], kind="stable")[:n]].tolist())
        b = set(both[np.argsort(-occl[both], kind="stable")[:n]].tolist())
        out["top5_jaccard"] = len(a & b) / len(a | b)
    return out


# ---------------------------------------------------------------- one fold


def run_cell(method: str, model_name: str, cell: int, cfg: dict, posts, *,
             batch_size: int, seed: int, limit: int | None, smoke: bool,
             allow: bool = False) -> None:
    from src.baselines import fine_tune as ft
    from src.stage2.blackbox import (ArmBox, GloVeBox, MentalBERTBox, ProbeBox,
                                     SAEBox, broadcast_to_tokens,
                                     text_word_deletion_rows,
                                     word_auprc_from_scores)
    from src.utils.dla import _word_plausibility
    from src.utils.sae_features import align_post_tokens, token_plausibility_auprc

    base = EXPLAINER_BASE[method]
    is_probe = group_of(base) == "probe"
    name = ARM_OF.get(method, method)
    max_len = int(cfg["full_finetune"]["max_len"])
    reduction = cfg["audit"]["word_score_aggregation"]
    k_grid = [float(k) for k in cfg["faithfulness"]["k_grid"]]

    eval_ids = span_eval_ids(posts, cfg, cell)
    sub = posts[posts["id"].isin(eval_ids)]
    if limit:
        sub = sub.head(limit)
    if base in SAE_BASES:
        box = SAEBox(base, model_name, cell, cfg, batch_size, seed, allow)
    elif base == "mentalbert":
        box = MentalBERTBox(base, model_name, cell, cfg, batch_size)
    elif base == "glove_bigru":
        box = GloVeBox(base, model_name, cell, cfg, batch_size)
    elif is_probe:
        box = ProbeBox(base, model_name, cell, cfg, batch_size, seed)
    else:
        box = ArmBox(base, model_name, cell, cfg, batch_size)
    if not box.ok:
        print(f"  {method} fold {cell}: no {base} checkpoint — SKIPPED "
              "(retain it with --keep-checkpoint; this never trains).", flush=True)
        return
    print(f"  {method} fold {cell}: {len(sub)} posts, {box.describe}", flush=True)
    text_only = getattr(box, "text_only", False)
    tok = None if text_only else box.tok

    plaus_rows, delete_rows, attrib_rows, eras_rows = [], [], [], []
    try:
        for n_done, (_, row) in enumerate(sub.iterrows(), start=1):
            text, pid = str(row["text"]), row["id"]
            span = (int(row["span_start"]), int(row["span_end"]))
            occl, clean = occlusion_scores(box, text)
            if text_only:
                # No study-backbone alignment: the cell is the occlusion
                # ranking scored on words, its deletion audit runs on text
                # with the model's own empty-post score, and its OWN scores
                # are the reference's mean-pooled DLA (src.stage2.reference_dla)
                # so the agreement columns are filled for it as well.
                from src.stage2.reference_dla import word_terms

                own = word_terms(box, base, text, cfg)[0]
                agree = agreement(own, occl)
                for i in np.flatnonzero(np.isfinite(own) | np.isfinite(occl)):
                    attrib_rows.append({"method": base, "model": model_name,
                                        "fold": cell, "id": pid, "word_index": int(i),
                                        "own": float(own[i]), "occlusion": float(occl[i])})
                plaus_rows.append({
                    "arm": name, "model": model_name, "fold": cell, "id": pid,
                    "token_auprc": float("nan"), "n_content_tokens": 0,
                    "n_gold_tokens": 0, "gold_base_rate": float("nan"),
                    **word_auprc_from_scores(text, occl, span),
                    "clean_logit": clean,
                    "decomposition_max_abs_error": float("nan"),
                    "truncated": False, "ranking_source": RANKING_SOURCE,
                    **agree})
                if np.isfinite(occl).any():
                    for r in text_word_deletion_rows(
                            box.score_texts, text, occl, config=cfg, seed=seed,
                            ranking_source=RANKING_SOURCE):
                        delete_rows.append({"arm": name, "model": model_name,
                                            "fold": cell, "id": pid, **r})
                if n_done == 1 or n_done % 20 == 0:
                    print(f"    {n_done}/{len(sub)}  occlusion word AUPRC so far "
                          f"{np.nanmean([p['word_auprc'] for p in plaus_rows]):.3f}",
                          flush=True)
                continue
            own = own_word_scores(box, base, text, pid, max_len=max_len,
                                  reduction=reduction)

            alignment = align_post_tokens(tok, text, max_len=max_len,
                                          span_start=span[0], span_end=span[1])
            ids, _, _ = ft.build_inputs(tok, [text], max_len)
            seq = broadcast_to_tokens(occl, alignment, ids.shape[1])
            n_content = int(alignment.token_indices.size)
            n_gold = int(alignment.gold_mask.sum())
            agree = agreement(own, occl)
            plaus_rows.append({
                "arm": name, "model": model_name, "fold": cell, "id": pid,
                "token_auprc": token_plausibility_auprc(seq, alignment),
                "n_content_tokens": n_content, "n_gold_tokens": n_gold,
                "gold_base_rate": (n_gold / n_content) if n_content else float("nan"),
                **_word_plausibility(seq, alignment, text, span[0], span[1], reduction),
                "clean_logit": clean,
                "decomposition_max_abs_error": float("nan"),
                "truncated": bool(alignment.truncated),
                "ranking_source": RANKING_SOURCE, **agree,
            })
            for i in np.flatnonzero(np.isfinite(own) | np.isfinite(occl)):
                attrib_rows.append({"method": base, "model": model_name,
                                    "fold": cell, "id": pid, "word_index": int(i),
                                    "own": float(own[i]), "occlusion": float(occl[i])})
            for r in eraser_rows(box, text, own, clean, k_grid):
                eras_rows.append({"method": base, "model": model_name, "fold": cell,
                                  "id": pid, "ranking": "own", **r})
            if np.isfinite(occl).any():
                if base in SAE_BASES:
                    delete_rows += box.deletion_rows(
                        text, occl, cfg=cfg, seed=seed, name=method,
                        model_name=model_name, cell=cell, pid=pid,
                        ranking_source=RANKING_SOURCE)
                elif is_probe:
                    from src.utils.frozen_backbone import probe_word_deletion_rows

                    delete_rows += probe_word_deletion_rows(
                        backbone=box.backbone, scorer=box.fitted.score_vectors,
                        text=text, word_scores=occl,
                        cached=box.ctx.acts[pid]["vecs"], config=cfg, seed=seed,
                        method=method, model=model_name, variant="", fold=cell,
                        pid=pid, ranking_source=RANKING_SOURCE)
                else:
                    from src.utils.faithfulness import arm_word_deletion_rows

                    for r in arm_word_deletion_rows(
                            box.adapter, tok, text, occl, config=cfg,
                            finetune_config=cfg["full_finetune"], seed=seed,
                            ranking_source=RANKING_SOURCE):
                        delete_rows.append({"arm": name, "model": model_name,
                                            "fold": cell, "id": pid, **r})
            if n_done == 1 or n_done % 20 == 0:
                print(f"    {n_done}/{len(sub)}  occlusion word AUPRC so far "
                      f"{np.nanmean([p['word_auprc'] for p in plaus_rows]):.3f}, "
                      f"spearman(own, occl) so far "
                      f"{np.nanmean([p['spearman'] for p in plaus_rows]):.2f}",
                      flush=True)
    finally:
        box.release()

    def write(stage, ident, rows):
        out = artifact_path(stage, ident, model_name, fold=cell, smoke=smoke)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(out)
        print(f"    wrote {out.name} ({len(rows)} rows)", flush=True)

    write("plaus", name, plaus_rows)
    write("attrib", base, attrib_rows)
    write("eraser", base, eras_rows)
    if is_probe:
        if smoke:
            write("delete", "probes", delete_rows)
        else:
            from src.stage2.word_deletion import _merge_probe_rows

            _merge_probe_rows(model_name, pd.DataFrame(delete_rows))
    else:
        write("delete", name, delete_rows)
    w = np.asarray([p["word_auprc"] for p in plaus_rows], dtype=float)
    s = np.asarray([p["spearman"] for p in plaus_rows], dtype=float)
    print(f"  {method} fold {cell}: occlusion word AUPRC {np.nanmean(w):.4f}, "
          f"spearman(own, occlusion) {np.nanmean(s):.3f} over {len(w)} posts",
          flush=True)


# ------------------------------------------------------------------ collect


def rbo(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> float:
    """Rank-biased overlap (Webber et al. 2010) of two rankings of the same
    items, extrapolated; top-weighted with persistence p."""
    depth = max(len(a), len(b))
    if depth == 0:
        return float("nan")
    seen_a, seen_b, overlap, total = set(), set(), 0, 0.0
    for d in range(1, depth + 1):
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        total += (overlap / d) * p ** (d - 1)
    return float((1 - p) * total + (overlap / depth) * p ** depth)


def top_weighted_agreement(model, base) -> dict | None:
    """Top-10% Jaccard and RBO between the own and the occlusion ranking, per
    post from the attrib_* rows, then mean and bootstrap interval."""
    files = sorted(globlib.glob(str(artifact_path("attrib", base, model, fold="*"))))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty or "id" not in df:          # text-only models write no own ranking
        return None
    jac, rbos = [], []
    for _pid, g in df.groupby("id"):
        both = g[np.isfinite(g["own"]) & np.isfinite(g["occlusion"])]
        n = len(both)
        if n < 5:
            continue
        k = max(1, int(np.ceil(0.10 * n)))
        own_rank = both.sort_values("own", ascending=False)["word_index"].to_numpy()
        occ_rank = both.sort_values("occlusion", ascending=False)["word_index"].to_numpy()
        a, b = set(own_rank[:k].tolist()), set(occ_rank[:k].tolist())
        jac.append(len(a & b) / len(a | b))
        rbos.append(rbo(own_rank, occ_rank))
    if not jac:
        return None
    j, jlo, jhi = mean_ci(jac)
    r, rlo, rhi = mean_ci(rbos)
    return {"top10_jaccard": j, "top10_jaccard_lo": jlo, "top10_jaccard_hi": jhi,
            "rbo": r, "rbo_lo": rlo, "rbo_hi": rhi, "n_posts_topk": len(jac)}


def collect(models) -> None:
    agree_rows, eras_rows = [], []
    for model in models:
        for method in METHODS:
            base = EXPLAINER_BASE[method]
            files = sorted(globlib.glob(str(artifact_path("plaus", ARM_OF.get(method, method),
                                                          model, fold="*"))))
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                rho, lo, hi = mean_ci(df["spearman"])
                jac, jlo, jhi = mean_ci(df["top5_jaccard"])
                agree_rows.append({
                    "method": base, "model": model, "attribution": RANKING_SOURCE,
                    "spearman": rho, "spearman_lo": lo, "spearman_hi": hi,
                    "spearman_median": float(df["spearman"].median()),
                    "top5_jaccard": jac, "top5_jaccard_lo": jlo, "top5_jaccard_hi": jhi,
                    **(top_weighted_agreement(model, base) or {}),
                    "n_posts": int(df["spearman"].notna().sum()),
                    "n_folds": int(df["fold"].nunique())})
                r = agree_rows[-1]
                print(f"  {model} {base:<22} spearman {rho:.3f} [{lo:.3f}, {hi:.3f}]  "
                      f"top-10% Jaccard {r.get('top10_jaccard', float('nan')):.3f}  "
                      f"RBO {r.get('rbo', float('nan')):.3f} ({r['n_posts']} posts)")
            files = sorted(globlib.glob(str(artifact_path("eraser", base, model, fold="*"))))
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                if df.empty or "id" not in df:      # text-only models: no own ranking
                    continue
                per_post = df.groupby("id")[["comprehensiveness", "sufficiency"]].mean()
                c, clo, chi = mean_ci(per_post["comprehensiveness"])
                s, slo, shi = mean_ci(per_post["sufficiency"])
                eras_rows.append({"method": base, "model": model, "ranking": "own",
                                  "comprehensiveness": c, "comprehensiveness_lo": clo,
                                  "comprehensiveness_hi": chi,
                                  "sufficiency": s, "sufficiency_lo": slo,
                                  "sufficiency_hi": shi,
                                  "n_posts": int(len(per_post))})
                for k, g in df.groupby("k_percent"):
                    eras_rows[-1][f"comp_k{int(k)}"] = float(g["comprehensiveness"].mean())
                    eras_rows[-1][f"suff_k{int(k)}"] = float(g["sufficiency"].mean())
                print(f"  {model} {base:<22} ERASER comprehensiveness {c:.3f} "
                      f"sufficiency {s:.3f} (prob space, mean over k)")
    if agree_rows:
        pd.DataFrame(agree_rows).to_csv(RESULTS / "attribution_agreement.csv", index=False)
        print(f"wrote attribution_agreement.csv ({len(agree_rows)} rows)")
    if eras_rows:
        pd.DataFrame(eras_rows).to_csv(RESULTS / "eraser.csv", index=False)
        print(f"wrote eraser.csv ({len(eras_rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect", action="store_true",
                    help="CPU: summarise the attrib_* and eraser_* rows")
    ap.add_argument("--allow-model-process", action="store_true")
    add_device_args(ap)
    args = ap.parse_args()
    if args.collect:
        models = [resolve_model(args.model)] if args.model else ["gemma-2-2b", "llama31-8b"]
        collect(models)
        return 0
    if args.limit and not args.smoke:
        raise SystemExit("--limit writes a partial cell; add --smoke")
    for m in args.method:
        assert CELLS_FROM[m] == "src.stage2.occlusion", m
    model = resolve_model(args.model)
    pin_visible_devices(args.devices, need_gpu=True,
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)
    cfg = load_config()
    require_model_process_approval(cfg, cli_override=args.allow_model_process,
                                   command="src.stage2.occlusion")
    from src.utils.data_prep import load_corpus

    posts, _ = load_corpus("lonxplain")
    print(f"occlusion — model {model}, methods {args.method}, folds {args.folds}")
    for method in args.method:
        for cell in args.folds:
            run_cell(method, model, cell, cfg, posts, batch_size=args.batch_size,
                     seed=args.seed, limit=args.limit, smoke=args.smoke,
                     allow=args.allow_model_process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
