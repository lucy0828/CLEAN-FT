"""DLA plausibility (word/token AUPRC vs psychiatrist spans) -> results/dla_auprc.csv.

Every method ranks each post's tokens by ITS OWN additive contribution to its
own logit, scored against the expert span: word AUPRC (pre-registered) with
token AUPRC beside it. Probes are refitted live on cached activations (CPU);
fine-tuning arms aggregate the per-post cells their checkpoints produced
(--compute scores missing cells from retained checkpoints; never trains).

How to run. --method defaults to every method, --model defaults to gemma2:
    uv run python -m src.stage2.dla_auprc                     # all methods, gemma2
    uv run python -m src.stage2.dla_auprc --model llama       # the family control
    uv run python -m src.stage2.dla_auprc --method linear_probe
    uv run python -m src.stage2.dla_auprc --method mlp_probe
    uv run python -m src.stage2.dla_auprc --method attention_probe
    uv run python -m src.stage2.dla_auprc --method sae_probe --allow-model-process
    uv run python -m src.stage2.dla_auprc --method full_ft lp_ft anchor
    uv run python -m src.stage2.dla_auprc --method attn_ft attn_lp_ft --compute
    uv run python -m src.stage2.dla_auprc --method full_ft --compute --allow-model-process
"""

import argparse
import glob as globlib
import sys

from src.utils.common import (
    ARM_OF,
    CELLS_FROM,
    DENSE_PROBES,
    PROBES,
    RESULTS,
    add_device_args,
    artifact_path,
    add_method_args,
    load_config,
    min_free_gb_for,
    pin_visible_devices,
    require_model_process_approval,
    resolve_model,
    selected_methods,
)
from src.utils.dla import (arm_row, dense_probe_rows, run_arm_cell,
                           sae_attention_probe_row, sae_probe_row)

# Every probe plus every fine-tuning arm. Taken from the registries rather
# than re-listed here: the attention-pooled arms were missing from a hand-kept
# copy of this tuple while their cells scored fine, so the list a method has to
# appear in is the same one `ARM_OF` dispatches on.
METHODS = PROBES + tuple(ARM_OF) + tuple(
    m for m in CELLS_FROM if m not in ARM_OF and m not in PROBES)

OUT_CSV = RESULTS / "dla_auprc.csv"

# --------------------------------------------------------------------- driver


def upsert(rows):
    """Merge `rows` into results/dla_auprc.csv on (method, model, variant).

    Serialized by an advisory lock, for the reason `record_decision` is:
    --compute cells are parallelized across GPUs, so several processes reach
    this function at once, and a read-modify-write of one CSV under concurrency
    loses whichever row was read before a neighbour's write landed. Holding the
    lock across the read AND the write makes the last writer the one with the
    most complete view of the cells on disk, rather than the one that happened
    to read earliest.
    """
    import fcntl

    import pandas as pd

    out = OUT_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = pd.DataFrame(rows)
        if out.exists():
            old = pd.read_csv(out).fillna({"variant": ""})
            keys = set(zip(table["method"], table["model"], table["variant"]))
            old = old[~old.apply(
                lambda r: (r["method"], r["model"], r["variant"]) in keys,
                axis=1)]
            table = pd.concat([old, table], ignore_index=True)
        table = table.sort_values(["model", "word_auprc"],
                                  ascending=[True, False])
        tmp = out.with_suffix(".csv.tmp")
        table.to_csv(tmp, index=False)
        tmp.replace(out)   # atomic: a reader never sees a half-written table
    print(f"\nwrote {out.name} ({len(table)} rows)")


def cell_row(method: str, model: str) -> dict | None:
    """The published row of a method whose plaus_* cells another module wrote
    (CELLS_FROM): read back exactly as `arm_row` reads an arm's, filed under
    the method's own name (its arm name where it has one)."""
    import glob as globlib

    import pandas as pd

    from src.utils.common import group_of, mean_ci

    name = ARM_OF.get(method, method)
    files = sorted(globlib.glob(str(artifact_path("plaus", name, model, fold="*"))))
    if not files:
        print(f"  {method:<20} no cells on disk — run {CELLS_FROM[method]}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w, w_lo, w_hi = mean_ci(df["word_auprc"])
    t, t_lo, t_hi = mean_ci(df["token_auprc"])
    return {
        "method": method, "group": group_of(method), "model": model,
        "variant": "",
        "word_auprc": w, "word_auprc_lo": w_lo, "word_auprc_hi": w_hi,
        "token_auprc": t, "token_auprc_lo": t_lo, "token_auprc_hi": t_hi,
        "n_posts": int(df["word_auprc"].notna().sum()),
        "n_folds": int(df["fold"].nunique()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Own-decomposition (DLA) word/token AUPRC vs psychiatrist "
                    "spans -> results/dla_auprc.csv")
    add_method_args(ap, choices=METHODS)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--word-aggregation", default=None,
                    choices=("max", "mean", "sum"),
                    help="sub-word -> word reduction (default: the "
                         "audit.word_score_aggregation pin)")
    ap.add_argument("--gpu", action="store_true",
                    help="let probe fitting use a GPU (default: pinned CPU)")
    ap.add_argument("--compute", action="store_true",
                    help="baselines: score missing arm cells from their "
                         "retained checkpoints before aggregating (GPU; "
                         "never trains)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="--compute: posts per forward pass (default 8)")
    add_device_args(ap)
    ap.add_argument("--allow-model-process", action="store_true")
    args = ap.parse_args()
    methods = selected_methods(args, choices=METHODS)
    model = resolve_model(args.model)
    pin_visible_devices(args.devices, need_gpu=bool(args.gpu or args.compute),
                        min_free_gb=min_free_gb_for(model),
                        wait_minutes=args.wait_for_gpu)

    from src.utils.common import load_config

    reduction = args.word_aggregation or load_config()["audit"]["word_score_aggregation"]

    print(f"DLA plausibility — model {model}, word reduction {reduction}")
    rows = []
    # Methods whose cells another module writes (CELLS_FROM): never refitted
    # or scored here, only read back -- probes and arms alike.
    external = [m for m in methods if m in CELLS_FROM]
    methods = [m for m in methods if m not in CELLS_FROM]
    dense = [m for m in methods if m in DENSE_PROBES]
    if dense:
        rows += dense_probe_rows(dense, model, args.folds, args.seed, reduction)
    if "sae_probe" in methods:
        rows.append(sae_probe_row(model, args.folds, args.seed,
                                  reduction, args.allow_model_process))
    if "sae_attention_probe" in methods:
        rows.append(sae_attention_probe_row(
            model, args.folds, args.seed, reduction,
            args.allow_model_process))
    arm_methods = [m for m in methods if m in ARM_OF]
    if args.compute and arm_methods:
        import glob as globlib

        from src.utils.data_prep import load_corpus

        require_model_process_approval(
            load_config(), cli_override=args.allow_model_process,
            command="src.stage2.dla_auprc --compute")
        lonxplain, _ = load_corpus("lonxplain")
        cfg = load_config()
        for m in arm_methods:
            arm = ARM_OF[m]
            cells = [0] if m == "anchor" else args.folds
            for cell in cells:
                if globlib.glob(str(
                        artifact_path("plaus", arm, model, fold=cell))):
                    continue
                run_arm_cell(arm, model, cell, cfg, lonxplain,
                             args.batch_size, reduction)
    for m in arm_methods:
        if (row := arm_row(m, model)) is not None:
            rows.append(row)
    for m in external:
        if (row := cell_row(m, model)) is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("no rows — see the messages above")
    for r in rows:
        print(f"  {r['method']:<20} {r['variant']:<24} "
              f"word {r['word_auprc']:.3f} [{r['word_auprc_lo']:.3f}, "
              f"{r['word_auprc_hi']:.3f}]  token {r['token_auprc']:.3f} "
              f"({r['n_posts']} posts)")
    upsert(rows)


if __name__ == "__main__":
    sys.exit(main())
