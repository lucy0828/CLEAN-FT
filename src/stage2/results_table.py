"""The consolidated results table -> RESULTS.md (\\section{Results}), from the csvs.

How to run (CPU; reads results/*.csv only):
    uv run python -m src.stage2.results_table            # install into RESULTS.md
    uv run python -m src.stage2.results_table --print    # print the LaTeX
    uv run python -m src.stage2.results_table --markdown # a markdown preview

ONE TABLE, LEFT TO RIGHT FROM DETECTION TO EVIDENCE
---------------------------------------------------
Each row is a method family at one depth; the two backbones are stacked as
row blocks. Columns: AUROC under mean pooling and under attention pooling
(the same read-out, one factor changed), then for the ATTENTION-pooled model
of the row its word AUPRC and its ΔAOPC under two attributions -- the model's
own read-out decomposition (DLA; Integrated Gradients in the pooled feature
space for the MLP probes) and one uniform input-level attribution applied
identically to every row (occlusion, `src.stage2.occlusion`). Every number
carries the half-width of the 2,000-draw post bootstrap interval the csv
carries, set small. Bold is the best value per column within a backbone
block over the method rows (floors and references excluded).

Sources: results/detection.csv (AUROC), results/dla_auprc.csv (word AUPRC),
results/word_deletion.csv (ΔAOPC margin), results/plausibility_floor.csv
(the random-ranking floor). The occlusion columns read the `occl_<method>`
rows those tables carry once `src.stage2.occlusion` has run. A cell whose
run has not landed is `--`; nothing is typed by hand.

INSTALL. RESULTS.md is the paper's LaTeX Results section. The table goes
between `% results-table:start` / `% results-table:end` markers; on first
install it replaces the detection table (`res:detection`) and removes the
two figure floats it supersedes (`fig:plausibility`, `fig:faithfulness`).
The prose is left as it is -- references to the removed figures are for the
author to rewrite, and the script prints where they are.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from src.utils.common import RESULTS, ROOT

MODELS = (("gemma-2-2b", "Gemma-2-2B"), ("llama31-8b", "Llama-3.1-8B"))

# (type, label, mean-pooled method, attention-pooled method, DLA note)
ROWS = (
    ("Fine-tune", "Full FT", "full_ft", "attn_ft", ""),
    ("", "ML-FT", "midlayer_ft", "midlayer_attn_ft", ""),
    ("", "LoRA-FT", "lora_ft", "attn_lora_ft", ""),
    ("Hybrid", "LP-FT", "lp_ft", "attn_lp_ft", ""),
    ("", "LP-ML-FT", "midlayer_lp_ft", "midlayer_attn_lp_ft", ""),
    ("Probe", "Linear", "linear_probe", "attention_probe", ""),
    ("", "MLP", "mlp_probe", "attention_mlp_probe", "IG"),
    # SAE: mean pooling only. The attention-pooled SAE cell is confounded by
    # the free query over 32,768 latents (see the shrunk-query arm), so its
    # attention AUROC and Δ are n/a and its evidence columns are the
    # mean-pooled probe's own decomposition and occlusion.
    ("", "SAE$^{\\ddagger}$", "sae_probe", None, "mean_only"),
    # Prompting sits inside each backbone block: one AUROC, no pooling, no
    # per-word score to attribute.
    ("Prompting", "--", "prompting", None, "prompt"),
)
# One reference block after both backbone blocks: the span-supervised
# fine-tune on each backbone (the evidence upper reference: it is trained on
# the expert spans, so it is never bolded), the two backbone-free encoders
# (detection only) and the model-free plausibility floors.
SPAN_REFERENCES = (("Reference", "Span-FT (Gemma)", "span_ft", "attn_span_ft", "gemma-2-2b"),
                   ("", "Span-FT (Llama)", "span_ft", "attn_span_ft", "llama31-8b"))
REFERENCES = (("", "MentalBERT", "mentalbert"),
              ("", "GloVe+BiGRU", "glove_bigru"))
FLOORS = (("Floor", "Random ranking", "random"),)
# The mean-pooled companion table: the same rows' MEAN-pooled models under
# their own decomposition, both backbones side by side.
MEAN_ROWS = tuple((typ, label, mean_m, note) for typ, label, mean_m, _a, note in ROWS
                  if note != "prompt") + (("Reference", "Span-FT", "span_ft", "ref"),)

START, END = "% results-table:start", "% results-table:end"
CAPTION = (
    "Detection and evidence on LonXplain. AUROC on the shared test set "
    "(1,761 posts) under mean and attention pooling, with $\\Delta$ their "
    "paired difference. For the attention-pooled model of each row: word "
    "AUPRC against the expert span on 499 posts under the model's own "
    "decomposition (DLA) and under occlusion, and $\\Delta$AOPC, the "
    "word-deletion margin of the model's own decomposition over a random "
    "control. Small numbers are 95\\% bootstrap half-widths; bold is the best "
    "per column within a backbone. $^{\\dagger}$Integrated Gradients in the "
    "pooled feature space. $^{\\ddagger}$Mean pooling only: the attention-pooled "
    "SAE cell is governed by its free query over 32,768 latents, so its "
    "attention cells are blank and its evidence columns are the mean-pooled "
    "probe's. Prompting yields one score per post. Span-FT is fine-tuned with the "
    "expert spans as extra supervision and is the evidence upper reference, so it "
    "is not bolded. MentalBERT and GloVe+BiGRU are mean-pooled with a linear head, "
    "so their DLA is the same decomposition; neither has an attention-pooled "
    "variant. The random floor is by construction except its measured AUPRC.")


def agreement_cell(attr, model):
    """(top-10% Jaccard of the DLA and occlusion rankings, half-width) from
    results/attribution_agreement.csv, or None."""
    path = RESULTS / "attribution_agreement.csv"
    if attr is None or not path.exists():
        return None
    t = pd.read_csv(path)
    hit = t[(t["method"] == attr) & (t["model"] == model)]
    if hit.empty or "top10_jaccard" not in hit or not np.isfinite(hit["top10_jaccard"].iloc[0]):
        return None
    r = hit.iloc[0]
    return float(r["top10_jaccard"]), (float(r["top10_jaccard_hi"]) - float(r["top10_jaccard_lo"])) / 2


def load():
    det = pd.read_csv(RESULTS / "detection.csv")
    dla = pd.read_csv(RESULTS / "dla_auprc.csv")
    dla["variant"] = dla["variant"].fillna("")
    dla = dla[~dla["variant"].str.contains("wordmean")]
    wd = pd.read_csv(RESULTS / "word_deletion.csv")
    floor = pd.read_csv(RESULTS / "plausibility_floor.csv")
    return det, dla, wd, floor


def paired_auroc_delta(attn_m, mean_m, model, det):
    """(attn - mean AUROC, half-width): the paired-bootstrap interval from
    results/paired_bootstrap.csv (src.stage2.uncertainty) when that run has
    landed, else the point difference from detection.csv with no interval."""
    if attn_m is None:
        return None
    path = RESULTS / "paired_bootstrap.csv"
    if path.exists():
        t = pd.read_csv(path)
        hit = t[(t["method_a"] == attn_m) & (t["method_b"] == mean_m)
                & (t["model"] == model) & (t["metric"] == "auroc")]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["diff"]), (float(r["diff_hi"]) - float(r["diff_lo"])) / 2
    a = cell(det, attn_m, model, "auroc", "auroc_lo", "auroc_hi")
    m = cell(det, mean_m, model, "auroc", "auroc_lo", "auroc_hi")
    if a is None or m is None:
        return None
    return a[0] - m[0], None


def cell(frame, method, model, col, lo, hi):
    if method is None:
        return None
    hit = frame[(frame["method"] == method) & (frame["model"] == model)]
    if hit.empty or not np.isfinite(hit[col].iloc[0]):
        return None
    v = float(hit[col].iloc[0])
    h = (float(hit[hi].iloc[0]) - float(hit[lo].iloc[0])) / 2
    return v, h


def rows_for(model, det, dla, wd):
    """[(kind, type, label, [6 cells], note)] for one backbone block."""
    values = []
    for typ, label, mean_m, attn_m, note in ROWS:
        attr = attn_m or mean_m            # the model the evidence columns read
        values.append(("method", typ, label, [
            cell(det, mean_m, model, "auroc", "auroc_lo", "auroc_hi"),
            cell(det, attn_m, model, "auroc", "auroc_lo", "auroc_hi"),
            paired_auroc_delta(attn_m, mean_m, model, det),
            cell(dla, attr, model, "word_auprc", "word_auprc_lo", "word_auprc_hi"),
            cell(dla, f"occl_{attr}", model, "word_auprc", "word_auprc_lo", "word_auprc_hi"),
            cell(wd, attr, model, "margin", "margin_lo", "margin_hi"),
        ], note))
    return values


def reference_rows(det, floor, dla=None, wd=None):
    out = []
    for typ, label, mean_m, attn_m, model in SPAN_REFERENCES:
        out.append(("ref", typ, label, [
            cell(det, mean_m, model, "auroc", "auroc_lo", "auroc_hi"),
            cell(det, attn_m, model, "auroc", "auroc_lo", "auroc_hi"),
            paired_auroc_delta(attn_m, mean_m, model, det),
            cell(dla, attn_m, model, "word_auprc", "word_auprc_lo", "word_auprc_hi"),
            cell(dla, f"occl_{attn_m}", model, "word_auprc", "word_auprc_lo", "word_auprc_hi"),
            cell(wd, attn_m, model, "margin", "margin_lo", "margin_hi"),
        ], ""))
    for typ, label, key in REFERENCES:
        hit = det[det["method"] == key]
        ref = None
        if len(hit):
            r = hit.iloc[0]
            ref = (float(r["auroc"]), (float(r["auroc_hi"]) - float(r["auroc_lo"])) / 2)
        # their occlusion cells are filed under the study default backbone
        # label (src/utils/common.py); the detector itself is backbone-free
        occl_auprc = cell(dla, f"occl_{key}", "gemma-2-2b", "word_auprc",
                          "word_auprc_lo", "word_auprc_hi")
        # their own decomposition (mean pooling, linear head: exact DLA) is
        # filed the same way by src.stage2.reference_dla
        dla_auprc = cell(dla, key, "gemma-2-2b", "word_auprc", "word_auprc_lo",
                         "word_auprc_hi")
        dla_aopc = cell(wd, key, "gemma-2-2b", "margin", "margin_lo", "margin_hi")
        out.append(("ref", typ, label, [ref, None, None, dla_auprc, occl_auprc, dla_aopc], ""))
    for typ, label, key in FLOORS:
        # model-free, identical on both backbones, quoted once. Its AUROC is
        # chance by construction and its ΔAOPC is the margin of a random
        # ranking over the random-word control -- zero in expectation -- so
        # those two are stated, not measured, and carry no interval.
        hit = floor[floor["method"] == key]
        auprc = None
        if len(hit):
            r = hit.iloc[0]
            auprc = (float(r["word_auprc"]),
                     (float(r["word_auprc_hi"]) - float(r["word_auprc_lo"])) / 2)
        chance_auroc, chance_aopc = (0.5, None), (0.0, None)
        out.append(("floor", typ, label,
                    [chance_auroc, chance_auroc, None, auprc, auprc, chance_aopc], ""))
    return out


def best_per_column(values):
    best = []
    for j in range(6):
        col = [v[3][j][0] for v in values
               if v[0] == "method" and v[4] != "prompt" and v[3][j] is not None]
        best.append(max(col) if col else None)
    return best


# ------------------------------------------------------------------ LaTeX


def _hw(h: float) -> str:
    """Half-width without its leading zero: .029, not 0.029."""
    return f"{h:.3f}".lstrip("0")


def tex_cell(c, bold, note_ig=False, signed=False):
    if c is None:
        return "--"
    v, h = c
    if signed:
        text = f"{v:+.3f}"
        return text + (f"{{\\scriptsize$\\pm${_hw(h)}}}" if h is not None else "")
    if h is None:                      # no interval
        if v in (0.0, 0.5):            # stated by construction
            return f"{v:.1f}"
        body = f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}"
        return body + ("$^{\\dagger}$" if note_ig else "")
    body = f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}"
    return body + f"{{\\scriptsize$\\pm${_hw(h)}}}" + ("$^{\\dagger}$" if note_ig else "")


def render_latex() -> str:
    det, dla, wd, floor = load()
    out = [START,
           "\\begin{table*}[t]", "\\centering",
           f"\\caption{{{CAPTION}}}", "\\label{res:main}", "\\small",
           "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{l l ccc cc c}", "\\toprule",
           "\\multirow{2}{*}{\\textbf{Type}} & \\multirow{2}{*}{\\textbf{Method}}"
           " & \\multicolumn{3}{c}{\\textbf{AUROC}}"
           " & \\multicolumn{2}{c}{\\textbf{AUPRC}}"
           " & \\textbf{$\\Delta$AOPC} \\\\",
           "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-8}",
           "& & Mean & Attn & $\\Delta$ & DLA & Occl. & DLA \\\\"]
    for model, name in MODELS:
        out += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{name}}}}} \\\\"]
        values = rows_for(model, det, dla, wd)
        best = best_per_column(values)
        for kind, typ, label, cells, note in values:
            cols = []
            for j, c in enumerate(cells):
                if note == "mean_only" and j in (1, 2):
                    cols.append("--")
                    continue
                bold = (j != 2 and note != "prompt" and c is not None
                        and best[j] is not None and abs(c[0] - best[j]) < 1e-12)
                cols.append(tex_cell(c, bold, signed=(j == 2),
                                     note_ig=(note == "IG" and j in (3, 5)
                                              and c is not None)))
            out.append(f"{typ} & {label} & " + " & ".join(cols) + " \\\\")
    out.append("\\midrule")
    for kind, typ, label, cells, note in reference_rows(det, floor, dla, wd):
        out.append(f"{typ} & {label} & "
                   + " & ".join(tex_cell(c, False, signed=(j == 2)) for j, c in enumerate(cells))
                   + " \\\\")
    # the mean-pooled companion table lives in APPENDIX.md
    # (src.stage2.appendix_tables imports mean_pooled_latex)
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", END]
    return "\n".join(out) + "\n"


MEAN_CAPTION = (
    "AUPRC and $\\Delta$AOPC for mean-pooled model. $^{\\dagger}$Integrated "
    "Gradients for MLP. Span-FT is trained on the expert spans (upper reference, "
    "not bolded).")


def mean_pooled_latex(det, dla, wd) -> list[str]:
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{MEAN_CAPTION}}}",
           "\\label{res:mean}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{l l cc cc}", "\\toprule",
           "\\multirow{2}{*}{\\textbf{Type}} & \\multirow{2}{*}{\\textbf{Method}}"
           " & \\multicolumn{2}{c}{\\textbf{Gemma-2-2B}}"
           " & \\multicolumn{2}{c}{\\textbf{Llama-3.1-8B}} \\\\",
           "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
           "& & AUPRC & $\\Delta$AOPC & AUPRC & $\\Delta$AOPC \\\\", "\\midrule"]
    grid = []
    for typ, label, mean_m, note in MEAN_ROWS:
        cells = []
        for model, _name in MODELS:
            cells.append(cell(dla, mean_m, model, "word_auprc", "word_auprc_lo", "word_auprc_hi"))
            cells.append(cell(wd, mean_m, model, "margin", "margin_lo", "margin_hi"))
        grid.append((typ, label, cells, note))
    best = []
    for j in range(4):
        col = [g[2][j][0] for g in grid if g[2][j] is not None and g[3] != "ref"]
        best.append(max(col) if col else None)
    for typ, label, cells, note in grid:
        # point estimates only in this table (the intervals live in the csvs)
        if note == "ref":
            out.append("\\midrule")
        cols = [tex_cell((c[0], None) if c is not None else None,
                         note != "ref" and c is not None and best[j] is not None
                         and abs(c[0] - best[j]) < 1e-12,
                         note_ig=(note == "IG" and c is not None))
                for j, c in enumerate(cells)]
        out.append(f"{typ} & {label} & " + " & ".join(cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return out


# --------------------------------------------------------------- markdown


def render_markdown() -> str:
    det, dla, wd, floor = load()
    lines = ["| Type | Method | AUROC mean | AUROC attn | Δ | AUPRC DLA | AUPRC occl. | ΔAOPC DLA |",
             "|---|---|---|---|---|---|---|---|"]
    for model, name in MODELS:
        lines.append(f"| *{name}* | | | | | | | |")
        values = rows_for(model, det, dla, wd)
        best = best_per_column(values)
        for kind, typ, label, cells, note in values:
            cols = []
            for j, c in enumerate(cells):
                if c is None:
                    cols.append("—"); continue
                t = (f"{c[0]:+.3f}" + (f" ±{_hw(c[1])}" if c[1] is not None else "") if j == 2 else
                     f"{c[0]:.3f} ±{_hw(c[1])}" if c[1] is not None else f"{c[0]:.1f}")
                if best[j] is not None and abs(c[0] - best[j]) < 1e-12:
                    t = f"**{t}**"
                cols.append(t)
            lines.append(f"| {typ} | {label} | " + " | ".join(cols) + " |")
    for kind, typ, label, cells, note in reference_rows(det, floor, dla, wd):
        lines.append(f"| {typ} | {label} | " + " | ".join(
            (f"{c[0]:.3f} ±{c[1]:.3f}" if c and c[1] is not None else (f"{c[0]:.1f}" if c else "—")) for c in cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- install


def install(latex: str) -> None:
    path = ROOT / "RESULTS.md"
    text = path.read_text()
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n",
                      lambda _m: latex, text, flags=re.S)
    else:
        # replace the detection table float with the consolidated one
        pattern = r"\\begin\{table\}\[t\]\n\\centering\n\\caption\{Loneliness detection AUROC.*?\\end\{table\}\n"
        if re.search(pattern, text, flags=re.S):
            text = re.sub(pattern, lambda _m: latex, text, count=1, flags=re.S)
        else:
            text = text.replace("\\section{Results}\n", "\\section{Results}\n\n" + latex, 1)
        # drop the two figure floats the table supersedes
        for label in ("fig:plausibility", "fig:faithfulness"):
            text = re.sub(r"\\begin\{figure\}\[[^\]]*\]\n(?:(?!\\end\{figure\}).)*?"
                          + re.escape(f"\\label{{{label}}}") + r"\n.*?\\end\{figure\}\n\n?",
                          "", text, flags=re.S)
    path.write_text(text)
    print(f"wrote the consolidated table into {path.name}")
    for label in ("fig:plausibility", "fig:faithfulness", "res:detection"):
        for m in re.finditer(re.escape(label), text):
            line = text.count("\n", 0, m.start()) + 1
            print(f"  note: `{label}` is still referenced at line {line} -- rewrite the prose")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    if args.markdown:
        print(render_markdown())
        return 0
    latex = render_latex()
    if args.print:
        print(latex)
        return 0
    install(latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
