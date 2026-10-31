"""Shared figure styling for the result plotters.

The plotting code lives in plot/ and every figure it draws is written to
plot/figs/, so the scripts and the PNGs they produce stay separate.

`plot/plot_detection.py` (Stage 1 detection), `plot/plot_error_rates.py` (its
decision-side companion) and `plot/plot_stage2.py` (Stage 2 plausibility
AUPRC) draw different tables but
must look like one set of figures: the same palette, the same fonts, the same
family bands, the same method order and the same short method names. Those live
here so a colour, an axis position or a display name is changed in one place
rather than four.

`plot/plot_feature_distortion.py` keeps its own copy of the ink/paper pair --
it is a standalone motivation figure that shades running text with a diverging
colormap, and neither that colormap nor the annotation green it marks the
psychiatrist's span in has any other use in this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# Rendered figures, not code: plot/ holds the scripts, plot/figs/ the PNGs.
OUT = Path(__file__).resolve().parent / "figs"

# Tableau 10 (2016). Lighter variants are the same hue mixed with white; the
# shade of a bar is which model produced it (SHADE_OF below), so hue reads as
# the method family and shade as the model, in both figures.
PROBE = "#4E79A7"
PROBE_LIGHT = "#8CA8C6"
PROBE_PALE = "#C3D2E1"
FT = "#F28E2B"
FT_LIGHT = "#F7B675"
FT_PALE = "#FBDBB6"
PROMPT = "#59A14F"
PROMPT_LIGHT = "#93C28D"
PROMPT_PALE = "#C7E0C4"
REF = "#BAB0AC"
REF_LIGHT = "#D2CCC9"
REF_PALE = "#E8E5E3"
INK = "#1C1917"
PAPER = "#FFFFFF"
BAND_PROBE = "#4E79A714"
BAND_FT = "#F28E2B14"
BAND_PROMPT = "#59A14F14"
BAND_REF = "#BAB0AC28"
CHANCE = "#6F6A67"
ARIAL_DIR = Path("/usr/share/fonts/truetype/liberation2")

# Backbone keys as `model` carries them in results/*.csv, plus the two
# backbone-free rows that label themselves (src.stage1.detection).
BACKBONES = {"gemma-2-2b": "Gemma-2-2B", "llama31-8b": "Llama-3.1-8B"}
BACKBONE_FREE_MODELS = {"mentalbert", "glove-300d"}

# Every `model` value the tables carry, in bar order inside a method's group,
# with the shade that names it. The two open backbones are the pair the study
# runs everywhere. The two backbone-free detectors bring their own encoder; they
# are drawn as reference lines rather than bars, so each takes the darkest shade.
#
# The API backbone (configs/default.yaml `prompting_api`, --model opus) is not a
# published series. It returns a decision and no token distribution, so it can
# never appear in the AUROC panel, and an arm that shows up in one of the two
# published panels and not the other is a hole in the comparison rather than a
# result. The arm still runs; it is simply not drawn.
SERIES = {
    "gemma-2-2b": "Gemma-2-2B",
    "llama31-8b": "Llama-3.1-8B",
    "mentalbert": "MentalBERT",
    "glove-300d": "GloVe+BiGRU",
}
SHADE_OF = {"gemma-2-2b": 0, "llama31-8b": 1,
            "mentalbert": 0, "glove-300d": 0}

# Short axis labels for the method names the CLIs write.
DISPLAY = {
    "linear_probe": "Linear",
    "mlp_probe": "MLP",
    "attention_probe": "Attn-Linear",
    # The two derived attention arms name what CONSTRAINS the query, because
    # that is the only thing that separates them from Attn-Linear.
    "shrunk_attention_probe": "Attn-Linear\n(shrunk q)",
    "attention_mlp_probe": "Attn-MLP",
    "sae_probe": "SAE",
    "sae_attention_probe": "Attn-SAE",
    "afr_probe": "AFR",
    "full_ft": "Full FT",
    "lp_ft": "LP-FT",
    "attn_ft": "Attn-FT",
    "attn_lp_ft": "Attn-LP-FT",
    # The mid-layer-headed arms read their own site, so they are the same
    # recipe with the ML- infix rather than differently named arms.
    "midlayer_ft": "ML-FT",
    "midlayer_attn_ft": "Attn-ML-FT",
    "midlayer_lp_ft": "LP-ML-FT",
    "midlayer_attn_lp_ft": "Attn-LP-\nML-FT",
    # The LoRA column: the same 2x2 with adapters in place of full updates.
    "lora_ft": "LoRA-FT",
    "lora_lp_ft": "LoRA-\nLP-FT",
    "attn_lora_ft": "Attn-\nLoRA-FT",
    "attn_lora_lp_ft": "Attn-LoRA-\nLP-FT",
    # Span-supervised: full_ft / attn_ft with the expert span in the loss.
    "span_ft": "Span-FT",
    "attn_span_ft": "Attn-\nSpan-FT",
    # Post-hoc explainers over the Full FT classifier.
    "gxi_ft": "Grad\u00d7Input\n(Full FT)",
    "occl_attn_ft": "Occlusion\n(Attn-FT)",
    "occl_midlayer_attn_ft": "Occlusion\n(Attn-ML-FT)",
    "occl_midlayer_attn_lp_ft": "Occlusion\n(Attn-LP-ML-FT)",
    "occl_attention_probe": "Occlusion\n(Attn-Linear)",
    "final_linear_probe": "Linear\n(final layer)",
    "final_attention_probe": "Attn-Linear\n(final layer)",
    "occl_attn_lp_ft": "Occlusion\n(Attn-LP-FT)",
    "occl_final_attention_probe": "Occlusion\n(Attn-Linear final)",
    "occl_last_token_probe": "Occlusion\n(Last-tok)",
    "occl_attention_mlp_probe": "Occlusion\n(Attn-MLP)",
    "occl_sae_attention_probe": "Occlusion\n(Attn-SAE)",
    "occl_sae_probe": "Occlusion\n(SAE)",
    "occl_mentalbert": "Occlusion\n(MentalBERT)",
    "occl_glove_bigru": "Occlusion\n(GloVe+BiGRU)",
    # Last-token read-outs (2026-09-09): the decoder-only default.
    "last_token_probe": "Last-tok\nLinear",
    "last_ft": "Last-tok\nFT",
    "anchor": "Anchor",
    "mentalbert": "Mental\nBERT",
    "glove_bigru": "GloVe+\nBiGRU",
    "prompting": "Prompt",
}

# Colour family. Coarser than the `group` column the tables carry, which only
# splits probe from baseline: here the fine-tunes, the prompt and the two
# reference detectors each get their own band.
FAMILY = {
    "linear_probe": "probe",
    "mlp_probe": "probe",
    "attention_probe": "probe",
    "shrunk_attention_probe": "probe",
    "attention_mlp_probe": "probe",
    "sae_probe": "probe",
    "sae_attention_probe": "probe",
    "afr_probe": "probe",
    "last_token_probe": "probe",
    "occl_attention_probe": "probe",
    "final_linear_probe": "probe",
    "final_attention_probe": "probe",
    "occl_final_attention_probe": "probe",
    "occl_last_token_probe": "probe",
    "occl_attention_mlp_probe": "probe",
    "occl_sae_attention_probe": "probe",
    "occl_sae_probe": "probe",
    "occl_attn_lp_ft": "ft",
    "full_ft": "ft",
    "lp_ft": "ft",
    "attn_ft": "ft",
    "attn_lp_ft": "ft",
    "midlayer_ft": "ft",
    "midlayer_attn_ft": "ft",
    "midlayer_lp_ft": "ft",
    "midlayer_attn_lp_ft": "ft",
    "lora_ft": "ft",
    "lora_lp_ft": "ft",
    "attn_lora_ft": "ft",
    "attn_lora_lp_ft": "ft",
    "span_ft": "ft",
    "attn_span_ft": "ft",
    "gxi_ft": "ft",
    "occl_attn_ft": "ft",
    "occl_midlayer_attn_ft": "ft",
    "occl_midlayer_attn_lp_ft": "ft",
    "last_ft": "ft",
    "anchor": "ft",
    "prompting": "prompt",
    "mentalbert": "ref",
    "glove_bigru": "ref",
    "occl_mentalbert": "ref",
    "occl_glove_bigru": "ref",
}

FAMILY_LABEL = {
    "probe": "Probe",
    "ft": "Fine-tune",
    "prompt": "Prompting",
    "ref": "Reference",
}

FAMILY_BAND = {
    "probe": BAND_PROBE,
    "ft": BAND_FT,
    "prompt": BAND_PROMPT,
    "ref": BAND_REF,
}


def apply_style() -> None:
    """Register Liberation Sans (metric-compatible with Arial) and set rcParams."""
    for fname in (
        "LiberationSans-Regular.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-BoldItalic.ttf",
    ):
        path = ARIAL_DIR / fname
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.formatter.use_mathtext": False,
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
            "axes.spines.top": False,
            "axes.grid": False,
        }
    )


def read_table(name: str, model: str | None = None, *, required: bool = True):
    """results/<name> as a DataFrame, with `variant` filled and rows for `model`.

    Every published table carries method/model/variant, and `variant` reads back
    as NaN for the methods that publish a single row, so it is filled here once
    rather than at each call site. `required=False` returns None for a table
    whose CLI has not been run yet, which is how the optional SAE-deletion
    series is skipped instead of drawn empty.
    """
    import pandas as pd

    path = RESULTS / name
    if not path.exists():
        if required:
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} — run the metric CLI that "
                "writes it first")
        return None
    table = pd.read_csv(path)
    if "variant" not in table.columns:
        table["variant"] = ""
    table["variant"] = table["variant"].fillna("")
    if model is not None:
        table = table[table["model"] == model]
    return table


def label_of(method: str, variant: str = "") -> str:
    """Axis label for one published row. A variant is its own tick."""
    base = DISPLAY.get(method, method)
    return f"{base}\n({variant})" if variant else base


def family_of(method: str) -> str:
    return FAMILY.get(method, "ref")


SHADES = {
    "probe": (PROBE, PROBE_LIGHT, PROBE_PALE),
    "ft": (FT, FT_LIGHT, FT_PALE),
    "prompt": (PROMPT, PROMPT_LIGHT, PROMPT_PALE),
    "ref": (REF, REF_LIGHT, REF_PALE),
}


def family_color(method: str, shade: int = 0) -> str:
    """The method's hue at one model's shade (0 darkest)."""
    tones = SHADES[family_of(method)]
    return tones[min(int(shade), len(tones) - 1)]


def model_color(method: str, model: str) -> str:
    return family_color(method, SHADE_OF.get(model, 0))


def fade(colour: str, weight: float, toward: str = PAPER) -> str:
    """`colour` mixed `weight` of the way from `toward` to itself."""
    from matplotlib.colors import to_hex, to_rgb

    weight = min(max(float(weight), 0.0), 1.0)
    return to_hex(tuple(b + (a - b) * weight
                        for a, b in zip(to_rgb(colour), to_rgb(toward))))


# The two ends of the lightness ramp `shade` runs between, as mixes of the
# hue: the pale end is that fraction of the hue over the paper, the deep end
# that fraction of the way from the hue to the ink.
#
# Both are pushed as far as their own constraint allows and no further. PALE is
# floored by the point LABELS, which take their cell's shade too and stop being
# readable as grey-blue type below about half strength. DEEP is capped by the
# HUE: past roughly two fifths of the way to the ink a dark blue, a dark brown
# and a dark teal stop being told apart, and hue is the primary channel here --
# a lightness ramp that eats the readout is a bad trade whatever it buys.
SHADE_PALE = 0.48
SHADE_DEEP = 0.42


def shade(colour: str, weight: float) -> str:
    """`colour` at `weight` of the ramp, 0 palest and 1 deepest.

    Lightness as a third number on a plane whose hue and shape are already
    taken (plot/plane.py's `shade_of`). The ramp runs from a WASH of the hue to
    a DARKENED one rather than from the hue to white: fading alone spans only
    the light half of the scale, and over the tenth of an AUROC the study's
    arms occupy that came out as one shade with a rounding error. Going both
    ways doubles the span for the same hue.

    It returns an opaque colour rather than setting alpha on the artist, and
    that is the point: two half-transparent markers that overlap composite to
    something darker than either, which on a figure where dark MEANS a higher
    score would draw a cell that does not exist.
    """
    pale = fade(colour, SHADE_PALE)
    deep = fade(INK, SHADE_DEEP, toward=colour)
    return fade(deep, weight, toward=pale)


# Left-to-right on both figures: probes, then the fine-tuning arms, then
# prompting, then the backbone-free reference detectors. A method the tables
# carry that is not named here is appended inside its family block, so a new
# method appears on the axis without editing this list.
METHOD_ORDER = [
    "linear_probe",
    "mlp_probe",
    "attention_probe",
    "shrunk_attention_probe",
    "attention_mlp_probe",
    "sae_probe",
    "sae_attention_probe",
    "afr_probe",
    "last_token_probe",
    "occl_attention_probe",
    "final_linear_probe",
    "final_attention_probe",
    "occl_final_attention_probe",
    "occl_last_token_probe",
    "occl_attention_mlp_probe",
    "occl_sae_attention_probe",
    "occl_sae_probe",
    "lp_ft",
    "full_ft",
    "attn_lp_ft",
    "attn_ft",
    "midlayer_ft",
    "midlayer_attn_ft",
    "midlayer_lp_ft",
    "midlayer_attn_lp_ft",
    "lora_ft",
    "lora_lp_ft",
    "attn_lora_ft",
    "attn_lora_lp_ft",
    "span_ft",
    "attn_span_ft",
    "gxi_ft",
    "occl_attn_ft",
    "occl_midlayer_attn_ft",
    "occl_midlayer_attn_lp_ft",
    "occl_attn_lp_ft",
    "last_ft",
    "anchor",
    "prompting",
]
FAMILY_ORDER = ["probe", "ft", "prompt", "ref"]


def order_key(method: str):
    """Sort key: family block first, METHOD_ORDER rank within it."""
    rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (FAMILY_ORDER.index(family_of(method)), rank, method)


def positions(methods):
    """Axis positions for `methods`, with a gap between families."""
    import numpy as np

    xs, x = [], 0.0
    for i, method in enumerate(methods):
        if i:
            if family_of(method) != family_of(methods[i - 1]):
                x += 0.55
            x += 1.0
        xs.append(x)
    return np.asarray(xs, dtype=float)


def group_bands(ax, methods, y_label, *, xs=None, pad=0.5, show_labels=True):
    """Shaded regions behind runs of same-family methods, labelled at y_label."""
    import numpy as np

    if xs is None:
        xs = np.arange(len(methods), dtype=float)
    i = 0
    while i < len(methods):
        fam = family_of(methods[i])
        j = i
        while j + 1 < len(methods) and family_of(methods[j + 1]) == fam:
            j += 1
        family_pad = 0.38 if fam == "ref" else pad
        left, right = xs[i] - family_pad, xs[j] + family_pad
        ax.axvspan(left, right, color=FAMILY_BAND[fam], zorder=0)
        if show_labels:
            ax.text(
                (left + right) / 2,
                y_label,
                FAMILY_LABEL[fam],
                ha="center",
                va="bottom",
                fontsize=14,
                color=family_color(methods[i]),
                fontstyle="italic",
            )
        i = j + 1


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------- readout x pooling encoding
# `plot/plane.py` draws the pooling axis as a visual variable rather than as a
# name in a label: HUE is the readout being pooled and SHAPE is the pooling.
# The axis is mean against learned; the max-pooled cells were removed from the
# study on 2026-08-27. The mapping lives here because it is the study's own
# factor structure (`src/probe/PROBES.md`), not one figure's private choice.
#
# A method belongs to a readout by what it reads and to a pooling by how it
# reduces the post's tokens. Fine-tuning arms carry a pooling too --
# `src/baselines/fine_tune.py` names it in every recipe -- so `full_ft` is a
# mean-pooled cell and `attn_ft` its attention-pooled partner.
READOUT = {
    "linear_probe": "linear",
    "attention_probe": "linear",
    # Both derived arms read the same linear direction; only the query differs,
    # so they are linear-readout cells and not a family of their own.
    "shrunk_attention_probe": "linear",
    "afr_probe": "linear",
    "mlp_probe": "mlp",
    "attention_mlp_probe": "mlp",
    "sae_probe": "sae",
    "sae_attention_probe": "sae",
    "full_ft": "ft",
    "lp_ft": "ft",
    "attn_ft": "ft",
    "attn_lp_ft": "ft",
    # The mid-layer column is its own hue rather than a marker on the FT hue:
    # the readout SITE is the factor those arms exist to vary, and drawing them
    # in fine-tune orange would put the study's site contrast inside a colour
    # two arms already share.
    "midlayer_ft": "ft_mid",
    "midlayer_attn_ft": "ft_mid",
    "midlayer_lp_ft": "ft_mid",
    "midlayer_attn_lp_ft": "ft_mid",
    # The LoRA column and the span arms are fine-tunes with a different
    # UPDATE (low-rank) or a different OBJECTIVE (span term); each gets its
    # own hue for the same reason the mid-layer column does.
    "lora_ft": "ft_lora",
    "lora_lp_ft": "ft_lora",
    "attn_lora_ft": "ft_lora",
    "attn_lora_lp_ft": "ft_lora",
    "span_ft": "ft_span",
    "attn_span_ft": "ft_span",
    # A post-hoc explainer is a different RANKING of the same classifier, not
    # a different fine-tune; its own hue keeps it from reading as one.
    "gxi_ft": "explainer",
    "occl_attn_ft": "explainer",
    "occl_midlayer_attn_ft": "explainer",
    "occl_midlayer_attn_lp_ft": "explainer",
    "occl_attention_probe": "explainer",
    "final_linear_probe": "linear",
    "final_attention_probe": "linear",
    "occl_attn_lp_ft": "explainer",
    "occl_final_attention_probe": "explainer",
    "occl_last_token_probe": "explainer",
    "occl_attention_mlp_probe": "explainer",
    "occl_sae_attention_probe": "explainer",
    "occl_sae_probe": "explainer",
    "occl_mentalbert": "explainer",
    "occl_glove_bigru": "explainer",
    "last_token_probe": "linear",
    "last_ft": "ft",
    "anchor": "ft",
}

POOLING = {
    "linear_probe": "mean",
    "attention_probe": "attn",
    "shrunk_attention_probe": "attn",
    "afr_probe": "mean",
    "mlp_probe": "mean",
    "attention_mlp_probe": "attn",
    "sae_probe": "mean",
    "sae_attention_probe": "attn",
    # From the recipe table in src/baselines/fine_tune.py: the full and LP arms
    # mean-pool and the two attn arms pool with a learned query.
    "full_ft": "mean",
    "lp_ft": "mean",
    "anchor": "mean",
    "attn_ft": "attn",
    "attn_lp_ft": "attn",
    "midlayer_ft": "mean",
    "midlayer_attn_ft": "attn",
    "midlayer_lp_ft": "mean",
    "midlayer_attn_lp_ft": "attn",
    "lora_ft": "mean",
    "lora_lp_ft": "mean",
    "attn_lora_ft": "attn",
    "attn_lora_lp_ft": "attn",
    "span_ft": "mean",
    "attn_span_ft": "attn",
    "gxi_ft": "mean",
    "occl_attn_ft": "attn",
    "occl_midlayer_attn_ft": "attn",
    "occl_midlayer_attn_lp_ft": "attn",
    "occl_attention_probe": "attn",
    "final_linear_probe": "mean",
    "final_attention_probe": "attn",
    "occl_attn_lp_ft": "attn",
    "occl_final_attention_probe": "attn",
    "occl_last_token_probe": "last",
    "occl_attention_mlp_probe": "attn",
    "occl_sae_attention_probe": "attn",
    "occl_sae_probe": "mean",
    "occl_mentalbert": "mean",
    "occl_glove_bigru": "mean",
    "last_token_probe": "last",
    "last_ft": "last",
}

# One hue per BACKBONE, for a figure that puts both of them in a single panel
# and so has the colour channel free (plot/plot_auprc_vs_aopc.py). The study's
# own two working colours: they are the furthest apart of the palette in both
# hue and lightness, which is what a two-class channel wants, and they are
# distinguishable to a red-green colourblind reader. They are NOT the readout
# hues -- a figure colouring by backbone does not key methods by colour at all,
# it labels them -- so nothing is being said twice in two grammars.
BACKBONE_COLOR = {"gemma-2-2b": PROBE, "llama31-8b": FT}

# Four hues, one per readout. FT keeps the orange it carries in every other
# figure in the study; prompting's green and the reference grey are deliberately
# NOT reused here, so a rug on the axis cannot be mistaken for a readout.
READOUT_COLOR = {
    "linear": "#4E79A7",
    "mlp": "#B07AA1",
    "sae": "#76B7B2",
    "ft": FT,
    "ft_mid": "#9C5311",
    "ft_lora": "#E15759",
    "ft_span": "#59A14F",
    "explainer": "#7F7F7F",
}
# Named for the arms a reader will actually see under the hue, not for the
# abstract family: the last-layer fine-tune hue is the LP-FT pair once
# `full_ft` is drawn as the reference and `attn_ft` is out, and "ML-FT" is the
# mid-layer column's own short name, matching the point labels in plane.LABEL.
READOUT_LABEL = {"linear": "Linear probe", "mlp": "MLP probe",
                 "sae": "SAE probe", "ft": "LP-FT",
                 "ft_mid": "LP-ML-FT", "ft_lora": "LoRA-FT",
                 "ft_span": "Span-FT", "explainer": "Post-hoc explainer"}
READOUT_ORDER = ["linear", "mlp", "sae", "ft", "ft_mid", "ft_lora", "ft_span",
                 "explainer"]

POOLING_MARKER = {"mean": "^", "attn": "o", "last": "s"}
POOLING_LABEL = {"mean": "Mean pool", "attn": "Attention pool",
                 "last": "Last token"}
POOLING_ORDER = ["mean", "attn", "last"]

# Which cells the connecting line joins, in POOLING_ORDER. The line is a
# POOLING contrast, so only arms that differ from each other in the pooling and
# in NOTHING else may sit on one. The fine-tunes therefore give TWO series
# rather than one: the head initialization is a second factor
# (src/baselines/fine_tune.py's 2x2), so `full_ft -> attn_ft` is the
# random-head pair and `lp_ft -> attn_lp_ft` the probe-init pair. Joining
# `lp_ft` to `attn_ft` would cross both factors at once and draw a pooling
# change that did not happen on its own.
#
# A series is keyed by its own name and coloured by its members' readout, so
# two series can share a hue. Keys are not readouts and nothing should index
# READOUT_COLOR with one.
POOLING_SERIES = {
    "linear": ["linear_probe", "attention_probe"],
    "mlp": ["mlp_probe", "attention_mlp_probe"],
    "sae": ["sae_probe", "sae_attention_probe"],
    "ft_random_head": ["full_ft", "attn_ft"],
    "ft_probe_head": ["lp_ft", "attn_lp_ft"],
    # The mid-layer column repeats the same two pooling contrasts one site
    # down. `shrunk_attention_probe` is deliberately NOT on the linear series:
    # it differs from `attention_probe` in how the query is CONSTRAINED, not in
    # the pooling, so joining it to `linear_probe` would draw a pooling change
    # that did not happen.
    "ft_mid_random_head": ["midlayer_ft", "midlayer_attn_ft"],
    "ft_mid_probe_head": ["midlayer_lp_ft", "midlayer_attn_lp_ft"],
    "ft_lora_random_head": ["lora_ft", "attn_lora_ft"],
    "ft_lora_probe_head": ["lora_lp_ft", "attn_lora_lp_ft"],
    "ft_span_random_head": ["span_ft", "attn_span_ft"],
}


# The OTHER factor of the fine-tuning 2x2: how the classification head was
# initialized. `full_ft` and `attn_ft` start from a random head; `lp_ft` and
# `attn_lp_ft` start from the solved probe at the same site and pooling. A
# series here may therefore differ in the head initialization and in NOTHING
# else, exactly as POOLING_SERIES may differ only in the pooling -- pairing
# `full_ft` with `attn_lp_ft` would cross both factors at once.
#
# There are two series and not three because the mid-layer column keeps only
# its probe-init cells; the mid-layer random-head pair was removed from the
# study on 2026-08-28, so `midlayer_lp_ft` has no partner to be compared with.
# The head-initialization factor: pairs that differ in whether the
# classification head starts from the solved probe at the same site, and in
# NOTHING else -- same pooling, same read-out site, same recipe. Grouped by the
# pooling each pair holds fixed.
#
# All four cells exist. The mid-layer random-head arms were removed on
# 2026-08-28 and restored on 2026-08-29 precisely because without them the
# mid-layer rows had no partner, and `midlayer_lp_ft` could only be compared
# against the FINAL-layer random head -- a comparison that moves the read-out
# site as well as the head, which is not this factor.
HEAD_INIT_SERIES = {
    "mean": {
        "Final layer": ["full_ft", "lp_ft"],
        "Mid-layer site": ["midlayer_ft", "midlayer_lp_ft"],
        "LoRA": ["lora_ft", "lora_lp_ft"],
    },
    "attn": {
        "Final layer": ["attn_ft", "attn_lp_ft"],
        "Mid-layer site": ["midlayer_attn_ft", "midlayer_attn_lp_ft"],
        "LoRA": ["attn_lora_ft", "attn_lora_lp_ft"],
    },
}
POOLING_PANEL_LABEL = {"mean": "Mean pooled", "attn": "Attention pooled"}
HEAD_INIT_ORDER = ["mean", "attn"]

# What the two ends of a HEAD_INIT_SERIES / POOLING_SERIES pair are called when
# a figure draws the factor as a before/after rather than as two named arms.
FACTOR_LEVELS = {
    "pooling": ("Mean pooling", "Attention pooling"),
}


def readout_of(method: str) -> str | None:
    return READOUT.get(method)


def pooling_of(method: str) -> str | None:
    return POOLING.get(method)


def readout_color(method: str) -> str:
    """The method's readout hue, or the neutral grey for an off-grid row."""
    return READOUT_COLOR.get(READOUT.get(method, ""), CHANCE)


# `src.utils.dla.word_variant` folds a non-pinned sub-word aggregation into the
# variant, so results/dla_auprc.csv carries a `--word-aggregation` sensitivity
# row BESIDE the published one for the same (method, backbone). Both figures
# that read the table would otherwise draw it: the plane joins on
# (method, backbone) and would plot the method twice at two heights, and the
# Stage 2 bars key on the variant and would grow an extra "wordmean" tick. The
# sensitivity belongs in the text, so one rule drops it for both.
WORD_SENSITIVITY = r"_word(?:max|mean|sum)$"


def drop_word_sensitivity(table, *, announce: bool = True):
    """Rows of `table` at the pinned sub-word aggregation only."""
    if table is None or "variant" not in table:
        return table
    flagged = table["variant"].str.contains(WORD_SENSITIVITY, regex=True,
                                            na=False)
    if announce:
        for _, row in table[flagged].iterrows():
            print(f"note: {row['method']} / {row['model']} variant "
                  f"{row['variant']} is a sub-word aggregation sensitivity "
                  "row — not drawn")
    return table[~flagged]
