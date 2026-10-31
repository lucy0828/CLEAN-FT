"""Stage 3, step 4: two clinicians' filled-in scripts -> one annotation table,
their spans located in the phase text, and their agreement.

How to run (CPU; the two filled-in copies of results/stage3/clinician_script.docx):
    uv run python -m src.stage3.annotator_labels \
        --annotator data/conversation/annotations/expert1.docx --annotator data/conversation/annotations/expert2.docx

Outputs:
    results/stage3/annotations.csv          one row per phase in the script,
                                            both annotators side by side
    results/stage3/annotator_agreement.csv  percent agreement, Cohen's kappa,
                                            Gwet's AC1, span overlap

THE SCHEMA KEEPS BOTH READINGS
------------------------------
Nothing is adjudicated. Each phase carries `annotator1_label` and
`annotator2_label` as two columns (1 = the checkbox was ticked, 0 = it was
not), each annotator's typed explanation verbatim, and each annotator's spans
located in the text as character ranges. A disputed phase stays disputed in
the table; how a disagreement is used is decided by the consumer
(`src.stage3.ood_detection --collect`, which evaluates against each annotator
and against a pooled view that keeps both readings of a disputed phase), not
here.

WHAT A SPAN IS
--------------
The script asked for an explanation under each phase, and the clinicians
answered by quoting the older adult -- several quotes for one phase separated
by a comma or a semicolon. Each quote is located in the phase's text
(`src.stage3.ood_corpus.load_phases`, the SAME string the detectors score) by
its normalised words: the longest window of the text whose word sequence
matches the quote's is taken, and a quote is accepted only when the match
covers at least `MATCH_FLOOR` of its words. A quote is read as a span only
under a ticked checkbox; the explanation under an unticked one is kept as
text and located nowhere, because it is a reason for a 0, not evidence for a
1. A quote that does not locate is reported by phase and left out rather than
guessed at.

Splitting a quote at its commas can only ever cut a contiguous span into
adjacent pieces that locate next to each other, so it changes nothing about
which WORDS are gold; it is what makes "two quotes separated by a comma" and
"one quote containing a comma" the same case.

AGREEMENT
---------
The two labellings are compared on every phase in the script. Percent
agreement is the raw proportion. Cohen's kappa corrects it by the agreement
two raters with these marginals would reach by chance; it is known to collapse
when one category dominates, which is the situation here (one annotator ticked
32 of 34), so Gwet's AC1 is reported beside it -- its chance term is built
from how evenly the two categories are used overall rather than from the
product of the raters' marginals, and it does not suffer that paradox. Both
carry a 2,000-draw phase bootstrap interval. Span overlap is reported on the
phases both annotators ticked and located at least one quote in: the word-level
Jaccard and F1 of the two gold sets, so a reader can see whether the two
clinicians who agree a phase is lonely also agree on where the evidence is.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.stage3.clinician_docx import collect
from src.stage3.ood_corpus import load_phases
from src.utils.common import ROOT, STAGE3, record_decision
from src.utils.data_prep import words

ANNOTATIONS_CSV = STAGE3 / "annotations.csv"
AGREEMENT_CSV = STAGE3 / "annotator_agreement.csv"

# A quote is accepted when this fraction of its words is found, in order, in
# the best window of the phase text. Transcripts and typed quotes differ by
# the odd dropped filler or corrected typo; a quote that misses more than a
# quarter of its words is not the transcript being quoted.
MATCH_FLOOR = 0.75
# Explanations list quotes separated by a semicolon or a comma (see the module
# docstring for why a comma inside a quote is harmless).
QUOTE_SEPARATOR = re.compile(r"[;,\n]+")

N_BOOT = 2000
SEED = 0


# ============================================================ locating spans


def quotes_of(explanation: str) -> list[str]:
    return [q.strip() for q in QUOTE_SEPARATOR.split(explanation or "")
            if q.strip()]


def locate(quote: str, text: str, text_words: list[dict],
           min_start: int = 0) -> tuple[int, int, float] | None:
    """(char_start, char_end, coverage) of `quote` in `text`, or None.

    Both sides are reduced to their normalised words (`data_prep.words`:
    lowercased, outer punctuation stripped). Every window of the text whose
    length is within two words of the quote's is compared to the quote by
    `difflib.SequenceMatcher`; the window with the most matched words wins,
    is trimmed to its first and last matched word, and is accepted when the
    matched words cover at least MATCH_FLOOR of the quote. `min_start` limits
    the search to windows beginning at or after that character position,
    which is how a sentence quoted twice is located twice rather than at its
    first occurrence both times.
    """
    q = [w["norm"] for w in words(quote) if w["norm"]]
    kept = [w for w in text_words if w["norm"] and w["start"] >= min_start]
    t = [w["norm"] for w in kept]
    if not q or not t:
        return None
    best = None
    for width in range(max(1, len(q) - 2), len(q) + 3):
        for start in range(0, max(1, len(t) - width + 1)):
            window = t[start:start + width]
            matcher = difflib.SequenceMatcher(None, window, q, autojunk=False)
            blocks = [b for b in matcher.get_matching_blocks() if b.size]
            matched = sum(b.size for b in blocks)
            if not blocks:
                continue
            first = start + blocks[0].a
            last = start + blocks[-1].a + blocks[-1].size - 1
            key = (matched, -(last - first))
            if best is None or key > best[0]:
                best = (key, first, last, matched)
    if best is None:
        return None
    _, first, last, matched = best
    coverage = matched / len(q)
    if coverage < MATCH_FLOOR:
        return None
    return int(kept[first]["start"]), int(kept[last]["end"]), float(coverage)


def _locate_in_order(quote: str, text: str, text_words: list[dict],
                     cursor: int) -> tuple[int, int, float] | None:
    """`locate`, first at or after `cursor`, then anywhere."""
    hit = locate(quote, text, text_words, min_start=cursor)
    if hit is None and cursor > 0:
        hit = locate(quote, text, text_words)
    return hit


# A quote that does not locate whole is retried one sentence at a time (see
# `locate_all`); a fragment with fewer than this many content words is not
# located on its own, because "Yes." matches anywhere.
MIN_FRAGMENT_WORDS = 2
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def locate_all(explanation: str, text: str) -> tuple[list[list[int]], list[str]]:
    """([[start, end], ...] for every located quote, [quotes that did not]).

    Quotes are located in the order they were typed, each searched from the
    end of the previous one first: annotators quote the transcript top to
    bottom, and a sentence the older adult said twice is then found twice. A
    quote that fails whole -- the annotator stitched two non-adjacent
    sentences, or left out a clause in the middle -- is retried sentence by
    sentence, so what they DID quote is kept and only the fragment that
    matches nothing is reported.
    """
    text_words = words(text)
    ranges, missed = [], []
    cursor = 0
    for quote in quotes_of(explanation):
        hit = _locate_in_order(quote, text, text_words, cursor)
        if hit is not None:
            ranges.append([hit[0], hit[1]])
            cursor = hit[1]
            continue
        fragments = [f.strip() for f in SENTENCE_BOUNDARY.split(quote)
                     if len([w for w in words(f) if w["norm"]])
                     >= MIN_FRAGMENT_WORDS]
        if len(fragments) < 2:
            missed.append(quote)
            continue
        for fragment in fragments:
            hit = _locate_in_order(fragment, text, text_words, cursor)
            if hit is None:
                missed.append(fragment)
            else:
                ranges.append([hit[0], hit[1]])
                cursor = hit[1]
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged, missed


def gold_words(text: str, ranges: list[list[int]]) -> set[int]:
    """Indices into `words(text)` of the content words any range overlaps."""
    out = set()
    for i, w in enumerate(words(text)):
        if not w["norm"]:
            continue
        if any(w["start"] < end and w["end"] > start for start, end in ranges):
            out.add(i)
    return out


# ================================================================= agreement


def percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1 for two raters over a binary item set.

    pa is the observed agreement. The chance term uses pi_k, the share of ALL
    ratings (both raters pooled) that fall in category k: pe = sum_k pi_k
    (1 - pi_k) / (K - 1), with K = 2 here. AC1 = (pa - pe) / (1 - pe).
    """
    pa = percent_agreement(a, b)
    ratings = np.concatenate([a, b])
    pi = np.array([(ratings == k).mean() for k in (0, 1)])
    pe = float((pi * (1 - pi)).sum())          # / (K - 1) with K = 2
    return float((pa - pe) / (1 - pe)) if pe < 1 else float("nan")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(np.unique(np.concatenate([a, b]))) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def bootstrap_ci(stat, a: np.ndarray, b: np.ndarray):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(a), len(a))
        values.append(stat(a[idx], b[idx]))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def span_overlap(table: pd.DataFrame) -> dict:
    """Word-level Jaccard and F1 between the two gold sets, where both exist."""
    jaccard, f1 = [], []
    for row in table.itertuples():
        if not (row.annotator1_label == 1 and row.annotator2_label == 1):
            continue
        g1 = gold_words(row.text, json.loads(row.annotator1_span_chars))
        g2 = gold_words(row.text, json.loads(row.annotator2_span_chars))
        if not g1 or not g2:
            continue
        inter, union = len(g1 & g2), len(g1 | g2)
        jaccard.append(inter / union)
        f1.append(2 * inter / (len(g1) + len(g2)))
    return {"span_jaccard": float(np.mean(jaccard)) if jaccard else float("nan"),
            "span_f1": float(np.mean(f1)) if f1 else float("nan"),
            "n_phases_both_spanned": len(jaccard)}


def agreement_table(table: pd.DataFrame) -> pd.DataFrame:
    a = table["annotator1_label"].to_numpy(dtype=int)
    b = table["annotator2_label"].to_numpy(dtype=int)
    rows = []
    for name, stat in (("percent_agreement", percent_agreement),
                       ("cohen_kappa", cohen_kappa),
                       ("gwet_ac1", gwet_ac1)):
        lo, hi = bootstrap_ci(stat, a, b)
        rows.append({"statistic": name, "value": stat(a, b), "lo": lo, "hi": hi,
                     "n_phases": len(a)})
    overlap = span_overlap(table)
    for name in ("span_jaccard", "span_f1"):
        rows.append({"statistic": name, "value": overlap[name],
                     "lo": float("nan"), "hi": float("nan"),
                     "n_phases": overlap["n_phases_both_spanned"]})
    both = int(((a == 1) & (b == 1)).sum())
    neither = int(((a == 0) & (b == 0)).sum())
    rows.append({"statistic": "n_both_lonely", "value": both,
                 "lo": float("nan"), "hi": float("nan"), "n_phases": len(a)})
    rows.append({"statistic": "n_neither_lonely", "value": neither,
                 "lo": float("nan"), "hi": float("nan"), "n_phases": len(a)})
    rows.append({"statistic": "n_disputed", "value": len(a) - both - neither,
                 "lo": float("nan"), "hi": float("nan"), "n_phases": len(a)})
    return pd.DataFrame(rows)


# ==================================================================== build


def build(docx_paths: list[str]) -> tuple[pd.DataFrame, dict]:
    if len(docx_paths) != 2:
        raise SystemExit("exactly two --annotator documents are expected")
    phases, _ = load_phases()
    text_of = phases.set_index("phase_id")["text"].to_dict()

    table = None
    missed_all = {}
    for n, path in enumerate(docx_paths, start=1):
        labels = collect(path)
        labels = labels[labels["clinician_lonely"].notna()].copy()
        col = f"annotator{n}"
        spans, missed = [], []
        for row in labels.itertuples():
            text = text_of.get(row.phase_id)
            if text is None:
                raise SystemExit(f"{row.phase_id} in {path} is not a screened "
                                 "phase of the current corpus")
            if int(row.clinician_lonely) == 1:
                ranges, miss = locate_all(row.clinician_explanation, text)
            else:
                ranges, miss = [], []
            spans.append(json.dumps(ranges))
            missed.append(miss)
            if miss:
                missed_all.setdefault(row.phase_id, {})[col] = miss
        part = pd.DataFrame({
            "phase_id": labels["phase_id"].to_numpy(),
            f"{col}_label": labels["clinician_lonely"].astype(int).to_numpy(),
            f"{col}_explanation": labels["clinician_explanation"].to_numpy(),
            f"{col}_span_chars": spans,
            f"{col}_n_spans": [len(json.loads(s)) for s in spans],
            f"{col}_unlocated_quotes": [len(m) for m in missed],
        })
        table = part if table is None else table.merge(part, on="phase_id",
                                                        how="outer")

    meta = phases[["phase_id", "corpus", "conversation_id", "type", "nickname",
                   "date", "n_user_words", "text"]]
    table = meta.merge(table, on="phase_id", how="inner")
    if table[["annotator1_label", "annotator2_label"]].isna().any().any():
        raise SystemExit("a phase is labelled by one annotator only; both "
                         "documents must carry every checkbox")
    table["annotator1_label"] = table["annotator1_label"].astype(int)
    table["annotator2_label"] = table["annotator2_label"].astype(int)
    table["agreed"] = table["annotator1_label"] == table["annotator2_label"]
    table = table.sort_values(["corpus", "nickname", "date", "phase_id"],
                              kind="stable").reset_index(drop=True)
    return table, missed_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotator", action="append", required=True,
                    metavar="DOCX",
                    help="a filled-in copy of the clinician script; give it "
                         "twice, first for annotator 1, then annotator 2")
    args = ap.parse_args()

    table, missed = build(args.annotator)
    STAGE3.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANNOTATIONS_CSV, index=False)
    agreement = agreement_table(table)
    agreement.to_csv(AGREEMENT_CSV, index=False)

    n = len(table)
    for k in (1, 2):
        pos = int(table[f"annotator{k}_label"].sum())
        spanned = int((table[f"annotator{k}_n_spans"] > 0).sum())
        print(f"annotator {k} ({args.annotator[k - 1]}): {pos} of {n} phases "
              f"lonely, {spanned} with at least one located span")
    if missed:
        print(f"  {sum(len(v) for m in missed.values() for v in m.values())} "
              f"quote(s) in {len(missed)} phase(s) could not be located in the "
              "older adult's text and are left out:")
        for pid, by in missed.items():
            for col, quotes in by.items():
                for q in quotes:
                    print(f"    {pid} {col}: {q[:90]!r}")
    print(f"wrote {ANNOTATIONS_CSV.relative_to(ROOT)} ({n} phases)")
    print(agreement.to_string(index=False))
    print(f"wrote {AGREEMENT_CSV.relative_to(ROOT)}")

    stats = agreement.set_index("statistic")["value"].to_dict()
    record_decision("stage3.annotator_labels", {
        "annotator1": str(args.annotator[0]),
        "annotator2": str(args.annotator[1]),
        "annotations_csv": str(ANNOTATIONS_CSV.relative_to(ROOT)),
        "n_phases": n,
        "annotator1_lonely": int(table["annotator1_label"].sum()),
        "annotator2_lonely": int(table["annotator2_label"].sum()),
        "n_disputed": int((~table["agreed"]).sum()),
        "percent_agreement": round(float(stats["percent_agreement"]), 4),
        "cohen_kappa": round(float(stats["cohen_kappa"]), 4),
        "gwet_ac1": round(float(stats["gwet_ac1"]), 4),
        "span_rule": "quotes split at ; and , and located by normalised-word "
                     f"window match, accepted at >= {MATCH_FLOOR} coverage; "
                     "read only under a ticked checkbox",
        "unlocated_quotes": {pid: by for pid, by in missed.items()},
        "adjudication": "none -- both labels kept as columns",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
