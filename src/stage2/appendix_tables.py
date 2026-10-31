"""The appendix tables -> APPENDIX.md (LaTeX), from the csvs.

How to run (CPU; reads results/*.csv only):
    uv run python -m src.stage2.appendix_tables            # rewrite APPENDIX.md
    uv run python -m src.stage2.appendix_tables --print

Everything between `% appendix-tables:start` / `:end` is regenerated; the
file is created with those markers if it does not exist. Tables, in order:

  app:mean      AUPRC / ΔAOPC of the mean-pooled models (src.stage2.results_table)
  app:agree     DLA-vs-occlusion agreement per attention-pooled model:
                top-10% Jaccard (fraction of top-decile words shared) and RBO,
                results/attribution_agreement.csv (src.stage2.occlusion --collect)

The seed, per-fold and OOD leave-one-participant-out tables (results/
seed_variation.csv, per_fold_auroc.csv, ood_clustered.csv) are kept as csvs
only; `seed_table`, `folds_table` and `lopo_table` below render them on
request but are not installed.

Missing cells print as `--`; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from src.stage2.results_table import _hw, load, mean_pooled_latex
from src.utils.common import RESULTS, ROOT

START, END = "% appendix-tables:start", "% appendix-tables:end"
MODELS = (("gemma-2-2b", "Gemma-2-2B"), ("llama31-8b", "Llama-3.1-8B"))
ATTN_ROWS = (("Fine-tune", "Full FT", "attn_ft"), ("", "ML-FT", "midlayer_attn_ft"),
             ("", "LoRA-FT", "attn_lora_ft"),
             ("Hybrid", "LP-FT", "attn_lp_ft"), ("", "LP-ML-FT", "midlayer_attn_lp_ft"),
             ("Probe", "Linear", "attention_probe"), ("", "MLP", "attention_mlp_probe"),
             ("", "SAE", "sae_probe"),
             ("Reference", "Span-FT", "attn_span_ft"))
MAIN_ROWS = (("Fine-tune", "Full FT", "full_ft", "attn_ft"),
             ("", "ML-FT", "midlayer_ft", "midlayer_attn_ft"),
             ("", "LoRA-FT", "lora_ft", "attn_lora_ft"),
             ("Hybrid", "LP-FT", "lp_ft", "attn_lp_ft"),
             ("", "LP-ML-FT", "midlayer_lp_ft", "midlayer_attn_lp_ft"),
             ("Probe", "Linear", "linear_probe", "attention_probe"),
             ("", "MLP", "mlp_probe", "attention_mlp_probe"),
             ("", "SAE", "sae_probe", None))


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def hit(frame, **keys):
    if frame is None:
        return None
    q = frame
    for k, v in keys.items():
        q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def num(v, h=None, digits=3):
    if v is None or not np.isfinite(v):
        return "--"
    text = f"{v:.{digits}f}"
    return text + (f"{{\\scriptsize$\\pm${_hw(h)}}}" if h is not None and np.isfinite(h) else "")


def table(caption, label, colspec, header_lines, body, small=True):
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}"]
    if small:
        out += ["\\small", "\\setlength{\\tabcolsep}{4pt}"]
    out += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header_lines, "\\midrule",
            *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return out


def agreement_table():
    t = read("attribution_agreement.csv")
    body = []
    for model, name in MODELS:
        body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for typ, label, m in ATTN_ROWS:
            r = hit(t, method=m, model=model)
            if r is None:
                cells = ["--", "--"]
            else:
                cells = [num(r.get("top10_jaccard", np.nan),
                             (r.get("top10_jaccard_hi", np.nan) - r.get("top10_jaccard_lo", np.nan)) / 2),
                         num(r.get("rbo", np.nan),
                             (r.get("rbo_hi", np.nan) - r.get("rbo_lo", np.nan)) / 2)]
            body.append(f"{typ} & {label} & " + " & ".join(cells) + " \\\\")
        if model != MODELS[-1][0]:
            body.append("\\midrule")
    body.append("\\midrule")
    for typ, label, m in (("Reference", "MentalBERT", "mentalbert"),
                          ("", "GloVe+BiGRU", "glove_bigru")):
        r = hit(t, method=m, model="gemma-2-2b")
        cells = (["--", "--"] if r is None else
                 [num(r.get("top10_jaccard", np.nan),
                      (r.get("top10_jaccard_hi", np.nan) - r.get("top10_jaccard_lo", np.nan)) / 2),
                  num(r.get("rbo", np.nan), (r.get("rbo_hi", np.nan) - r.get("rbo_lo", np.nan)) / 2)])
        body.append(f"{typ} & {label} & " + " & ".join(cells) + " \\\\")
    return table(
        "Agreement between each model's own decomposition (DLA) and occlusion, per "
        "post: the fraction of the top-decile words shared by the two rankings "
        "(top-10\\% Jaccard) and rank-biased overlap (RBO, $p=0.9$). Decoder rows "
        "are the attention-pooled models; the references are mean-pooled. Small "
        "numbers are bootstrap half-widths.",
        "app:agree", "l l cc",
        ["\\textbf{Type} & \\textbf{Method} & Top-10\\% Jaccard & RBO \\\\"],
        body)


def eraser_table():
    t = read("eraser.csv")
    body = []
    for model, name in MODELS:
        body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for typ, label, m in ATTN_ROWS:
            r = hit(t, method=m, model=model)
            cells = ([num(r["comprehensiveness"], (r["comprehensiveness_hi"] - r["comprehensiveness_lo"]) / 2),
                      num(r["sufficiency"], (r["sufficiency_hi"] - r["sufficiency_lo"]) / 2)]
                     if r is not None else ["--", "--"])
            body.append(f"{typ} & {label} & " + " & ".join(cells) + " \\\\")
        if model != MODELS[-1][0]:
            body.append("\\midrule")
    return table(
        "ERASER faithfulness in probability space for each model's own ranking, "
        "averaged over the $k$ grid: comprehensiveness $p(x)-p(x\\setminus\\text{top-}k)$ "
        "(higher is better) and sufficiency $p(x)-p(\\text{top-}k\\text{ only})$ (lower "
        "is better). Sufficiency keeps words rather than deleting them, so it does "
        "not favour an attribution fit to deletions.",
        "app:eraser", "l l cc",
        ["\\textbf{Type} & \\textbf{Method} & Comprehensiveness $\\uparrow$ & Sufficiency $\\downarrow$ \\\\"],
        body)


def seed_table():
    t = read("seed_variation.csv")
    labels = {"attention_probe": "Attn-Linear", "attn_ft": "Attn-FT",
              "midlayer_attn_lp_ft": "Attn-LP-ML-FT"}
    metrics = (("auroc", "AUROC"), ("word_auprc", "AUPRC"), ("aopc_margin", "$\\Delta$AOPC"))
    body = []
    for m, label in labels.items():
        for i, (metric, mname) in enumerate(metrics):
            r = hit(t, method=m, model="gemma-2-2b", metric=metric)
            if r is None:
                cells = ["--"] * 5
            else:
                cells = [num(r.get("seed0", np.nan)), num(r.get("seed1", np.nan)),
                         num(r.get("seed2", np.nan)), num(r["mean"]), num(r["sd"])]
            body.append(f"{label if i == 0 else ''} & {mname} & " + " & ".join(cells) + " \\\\")
    return table(
        "Seed-to-seed variation of the Gemma-2-2B headline cells. Seeds 1 and 2 re-run "
        "training (head initialisation and data order), the attention probe's query "
        "initialisation, and the deletion audit's random control; folds, span-eval "
        "posts and bootstrap draws are pinned.",
        "app:seeds", "l l ccc cc",
        ["\\textbf{Method} & \\textbf{Metric} & seed 0 & seed 1 & seed 2 & mean & SD \\\\"],
        body)


def folds_table():
    t = read("per_fold_auroc.csv")
    body = []
    for model, name in MODELS:
        body.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for typ, label, mean_m, attn_m in MAIN_ROWS:
            for pooling, m in (("mean", mean_m), ("attn", attn_m)):
                if m is None:
                    continue
                r = hit(t, method=m, model=model)
                cells = ([num(r[f"fold{k}"]) for k in (0, 2, 4, 6, 8)]
                         + [num(r["fold_mean"]), num(r["fold_sd"])]) if r is not None else ["--"] * 7
                body.append(f"{typ if pooling == 'mean' else ''} & {label} ({pooling}) & "
                            + " & ".join(cells) + " \\\\")
        if model != MODELS[-1][0]:
            body.append("\\midrule")
    return table(
        "Per-fold AUROC on the five held test folds, with their mean and SD. The main "
        "table's AUROC is the single pooled out-of-fold value.",
        "app:folds", "l l ccccc cc",
        ["\\textbf{Type} & \\textbf{Method} & f0 & f2 & f4 & f6 & f8 & mean & SD \\\\"],
        body)


def lopo_table():
    t = read("ood_clustered.csv")
    names = {"full_ft": "Full FT", "attn_lp_ft": "LP-FT*", "midlayer_attn_lp_ft": "LP-ML-FT*",
             "attention_probe": "Linear*", "attention_mlp_probe": "MLP*"}
    body = []
    for model, name in MODELS:
        body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for m, label in names.items():
            for view in ("annotator1", "annotator2"):
                r = hit(t, method=m, model=model, annotator=view)
                if r is None:
                    cells = ["--"] * 4
                else:
                    rng = lambda a, b: f"{a:.3f}--{b:.3f}" if np.isfinite(a) and np.isfinite(b) else "--"
                    cells = [f"{num(r['auroc_annotated'])} [{num(r['auroc_annotated_lo_cluster'])}, {num(r['auroc_annotated_hi_cluster'])}]",
                             rng(r["lopo_min"], r["lopo_max"]),
                             rng(r.get("auprc_lopo_min", np.nan), r.get("auprc_lopo_max", np.nan)),
                             rng(r.get("aopc_lopo_min", np.nan), r.get("aopc_lopo_max", np.nan))]
                body.append(f"{label if view == 'annotator1' else ''} & {'A1' if view == 'annotator1' else 'A2'} & "
                            + " & ".join(cells) + " \\\\")
        if model != MODELS[-1][0]:
            body.append("\\midrule")
    return table(
        "OOD robustness to participants. Expert-only AUROC on the 34 annotated phases "
        "with a participant-clustered bootstrap interval, and the range of each metric "
        "over leave-one-participant-out (seven participants; one contributes 16 of the "
        "34 phases). A1/A2: annotator.",
        "app:lopo", "l l l ccc",
        ["\\textbf{Method} & & AUROC [cluster 95\\%] & AUROC LOPO & AUPRC LOPO & $\\Delta$AOPC LOPO \\\\"],
        body)


def render() -> str:
    det, dla, wd, _floor = load()
    out = [START, "\\section{Additional results}", ""]
    mean = mean_pooled_latex(det, dla, wd)
    mean = [l.replace("\\label{res:mean}", "\\label{app:mean}") for l in mean]
    out += mean + [""]
    out += agreement_table() + [END]
    return "\n".join(out) + "\n"


def install(latex: str) -> None:
    path = ROOT / "APPENDIX.md"
    text = path.read_text() if path.exists() else ""
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n",
                      lambda _m: latex, text, flags=re.S)
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + latex
    path.write_text(text)
    print(f"wrote the appendix tables into {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    latex = render()
    if args.print:
        print(latex)
        return 0
    install(latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
