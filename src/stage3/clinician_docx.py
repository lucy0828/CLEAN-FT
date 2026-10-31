"""Stage 3, step 3: a fillable Word copy of the clinician script, and the
reverse pass that reads a filled-in copy back into a CSV of clinician labels.

results/stage3/clinician_script.md is what a clinician reads. This module
produces the SAME flagged phases as a compact .docx: nickname and
conversation id, the transcript with no timestamps, a real Word checkbox,
and a line for the clinician's own explanation -- no other headings, and
none of the chatbot's own backend artifacts (see STRIPPED SYSTEM ARTIFACTS
below), so a clinician can open it, read, check, type, and save, with no
other tooling.

Each phase carries two anchors, neither shown to the clinician as text:
  - the checkbox is a real Word checkbox content control (w14:checkbox,
    what Word's own Developer-tab "Check Box Content Control" inserts),
    tagged (w:tag) with that phase's phase_id -- this is how the collect
    pass finds its way back to the row a checkbox belongs to
  - the phase's own bold, 12pt nickname/conversation heading, which the
    collect pass recognises by its formatting (not its text) as the start
    of a new phase, so it knows where one phase's typed explanation ends
    and the next phase's begins
Nothing is re-derived from paragraph position or from any visible id.

STRIPPED SYSTEM ARTIFACTS
--------------------------
Some CHATBOT messages carry the app's own backend call leaking into the
transcript text -- either the natural-language sign-off ("Ending the
intervention now.") or the literal call itself ("end_intervention()",
"complete_questions", "complete_checkin"). Neither is something the chatbot
or the older adult actually said, so both are stripped from a message
before it is written to the .docx; a message that turns out to be nothing
but the artifact is dropped entirely. This never touches dialogue.csv or
clinician_script.md -- only what this module renders.

How to run (step 2, `src.stage3.detect_loneliness --allow-network`, must
already have written screened_phases_{corpus}.parquet for every corpus):
    uv run python -m src.stage3.clinician_docx --build
    uv run python -m src.stage3.clinician_docx --collect PATH_TO_FILLED_DOCX

Outputs:
    results/stage3/clinician_script.docx   the fillable document
    results/stage3/clinician_labels.csv    written by --collect: one row per
                                            flagged phase, with the model's
                                            screen, the clinician's checkbox
                                            state, and their typed explanation

A CAVEAT ON THE CHECKBOX
-------------------------
w14:checkbox is a Microsoft extension, not part of the base OOXML schema.
Real Microsoft Word reads and writes it natively -- this is exactly what its
own content-control checkbox produces -- but a document that gets opened and
re-saved by something other than Word (Google Docs, some other editor) can
flatten or drop it. Before sending this to a clinician, open the built file
in Word yourself and confirm the boxes are still checkable; --collect prints
a loud warning, and skips rows silently rather than guessing, if it finds
fewer checkboxes than there were flagged phases.

PHI: this file carries the same participant speech as clinician_script.md.
Nothing here sends it anywhere -- sharing the .docx is on the user.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib

import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from src.stage3.detect_loneliness import USER, load_dialogue, message_text, output_path
from src.stage3.prepare_conversations import CORPORA
from src.utils.common import ROOT, STAGE3, record_decision

DOCX_PATH = STAGE3 / "clinician_script.docx"
LABELS_CSV = STAGE3 / "clinician_labels.csv"

FONT = "Arial"
HEADER_SIZE = 12
BODY_SIZE = 9   # a bit smaller than HEADER_SIZE, for the transcript and labels
UNCHECKED_GLYPH = "☐"   # ballot box; Word swaps this for the checked glyph itself
EXPLANATION_LABEL = "Explanation:"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
W14_NS = "{http://schemas.microsoft.com/office/word/2010/wordml}"

# Natural-language sign-off ("Ending the intervention now." / "..., Stanley."
# -- including the dangling "Ending the" left when interleaved older-adult
# speech splits it from "intervention now.") or the literal backend call
# ("end_intervention()", "complete_questions", with or without the parens)
# -- both are the app's own state-machine leaking into what should be
# spoken text, on top of no real English word ever containing an underscore.
_SYSTEM_ARTIFACT_RE = re.compile(
    r"\s*(?:Ending the intervention now(?:,\s*\w+)?\.?"
    r"|Ending the\b"
    r"|\b[a-z]+(?:_[a-z]+)+(?:\(\))?)\s*",
    re.IGNORECASE,
)


def _strip_system_artifacts(text: str) -> str:
    return _SYSTEM_ARTIFACT_RE.sub(" ", text).strip()


# ========================================================== the checkbox sdt


def _add_checkbox(paragraph, tag_value: str) -> None:
    """Append an inline Word checkbox content control, tagged with tag_value.

    This is the w14:checkbox extension element, in the same shape Word's own
    Developer-tab checkbox control writes: an <w:sdt> whose <w:sdtPr> carries
    an id, the tag used for later lookup, and the checkbox definition itself;
    its content is the glyph run Word swaps between checked and unchecked.
    """
    cb_id = zlib.crc32(tag_value.encode("utf-8")) % 1_000_000_000

    sdt = OxmlElement("w:sdt")
    sdt_pr = OxmlElement("w:sdtPr")

    id_el = OxmlElement("w:id")
    id_el.set(qn("w:val"), str(cb_id))
    tag_el = OxmlElement("w:tag")
    tag_el.set(qn("w:val"), tag_value)

    checkbox = OxmlElement("w14:checkbox")
    checked = OxmlElement("w14:checked")
    checked.set(qn("w14:val"), "0")
    checked_state = OxmlElement("w14:checkedState")
    checked_state.set(qn("w14:val"), "2612")
    checked_state.set(qn("w14:font"), "MS Gothic")
    unchecked_state = OxmlElement("w14:uncheckedState")
    unchecked_state.set(qn("w14:val"), "2610")
    unchecked_state.set(qn("w14:font"), "MS Gothic")
    checkbox.append(checked)
    checkbox.append(checked_state)
    checkbox.append(unchecked_state)

    sdt_pr.append(id_el)
    sdt_pr.append(tag_el)
    sdt_pr.append(checkbox)

    sdt_content = OxmlElement("w:sdtContent")
    run = OxmlElement("w:r")
    run_pr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "MS Gothic")
    fonts.set(qn("w:eastAsia"), "MS Gothic")
    fonts.set(qn("w:hAnsi"), "MS Gothic")
    fonts.set(qn("w:hint"), "eastAsia")
    run_pr.append(fonts)
    text_el = OxmlElement("w:t")
    text_el.text = UNCHECKED_GLYPH
    run.append(run_pr)
    run.append(text_el)
    sdt_content.append(run)

    sdt.append(sdt_pr)
    sdt.append(OxmlElement("w:sdtEndPr"))
    sdt.append(sdt_content)
    paragraph._p.append(sdt)


def _iter_checkboxes(document: Document):
    """Every (tag, checked) pair from the w14 checkboxes in a document."""
    for sdt in document.element.body.iter(f"{W_NS}sdt"):
        tag = _sdt_checkbox_tag(sdt)
        if tag is None:
            continue
        sdt_pr = sdt.find(f"{W_NS}sdtPr")
        checkbox = sdt_pr.find(f"{W14_NS}checkbox")
        checked_el = checkbox.find(f"{W14_NS}checked")
        checked = checked_el is not None and checked_el.get(f"{W14_NS}val") == "1"
        yield tag, checked


def _sdt_checkbox_tag(sdt) -> str | None:
    sdt_pr = sdt.find(f"{W_NS}sdtPr")
    if sdt_pr is None:
        return None
    checkbox = sdt_pr.find(f"{W14_NS}checkbox")
    if checkbox is None:
        return None
    tag_el = sdt_pr.find(f"{W_NS}tag")
    if tag_el is None:
        return None
    return tag_el.get(f"{W_NS}val")


def _checkbox_tag_in_paragraph(paragraph) -> str | None:
    for sdt in paragraph._p.iter(f"{W_NS}sdt"):
        tag = _sdt_checkbox_tag(sdt)
        if tag is not None:
            return tag
    return None


def _is_header_paragraph(paragraph) -> bool:
    """True for the bold, HEADER_SIZE-pt "nickname — conversation id" line
    build_docx writes once per phase -- the structural start-of-phase marker
    the collect pass uses instead of any visible id."""
    if not paragraph.runs:
        return False
    run = paragraph.runs[0]
    return bool(run.bold) and run.font.size == Pt(HEADER_SIZE)


# ================================================================= building


def _paragraph(document: Document, text: str = "", *, bold: bool = False,
              size: int = BODY_SIZE, color: RGBColor | None = None):
    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    return p


def flagged_phases(corpus: str, dialogue: pd.DataFrame,
                   scored: pd.DataFrame) -> list[dict]:
    """One entry per flagged phase: nickname, conversation id, transcript."""
    flagged = {(row.conversation_id, row.type): row.phase_id
              for row in scored.itertuples() if int(row.lonely) == 1}
    if not flagged:
        return []
    scoped = dialogue[dialogue["conversation_id"].isin(
        {cid for cid, _ in flagged})]

    out = []
    for (cid, phase), rows in scoped.groupby(["conversation_id", "type"],
                                             sort=False):
        if (cid, phase) not in flagged:
            continue
        rows = rows.sort_values("order", kind="stable")
        messages = []
        for row in rows.itertuples():
            text = _strip_system_artifacts(message_text(row))
            if text:
                messages.append({"speaker": row.speaker, "text": text})
        out.append({
            "phase_id": flagged[(cid, phase)],
            "nickname": str(rows["nickname"].iloc[0]),
            "cid8": cid[:8],
            "date": str(rows["date"].iloc[0]),
            "messages": messages,
        })
    out.sort(key=lambda p: (p["nickname"], p["date"]))
    return out


def negative_phases(corpus: str, dialogue: pd.DataFrame, scored: pd.DataFrame,
                    n: int, rng) -> list[dict]:
    """`n` screen-NEGATIVE phases of `corpus` drawn at random, in the same
    shape `flagged_phases` returns, so the clinician reads them blind to the
    screener's verdict. The reviewer's point: with every negative
    screener-derived, the OOD AUROC's negatives are the screener's, not a
    clinician's; annotating a random sample of them puts a clinician's 0s
    beside the screener's."""
    negatives = [(row.conversation_id, row.type, row.phase_id)
                 for row in scored.itertuples() if int(row.lonely) == 0]
    if not negatives:
        return []
    pick = rng.choice(len(negatives), size=min(n, len(negatives)), replace=False)
    chosen = {(negatives[i][0], negatives[i][1]): negatives[i][2] for i in pick}
    scoped = dialogue[dialogue["conversation_id"].isin({cid for cid, _ in chosen})]
    out = []
    for (cid, phase), rows in scoped.groupby(["conversation_id", "type"],
                                             sort=False):
        if (cid, phase) not in chosen:
            continue
        rows = rows.sort_values("order", kind="stable")
        messages = []
        for row in rows.itertuples():
            text = _strip_system_artifacts(message_text(row))
            if text:
                messages.append({"speaker": row.speaker, "text": text})
        out.append({"phase_id": chosen[(cid, phase)],
                    "nickname": str(rows["nickname"].iloc[0]),
                    "cid8": cid[:8], "date": str(rows["date"].iloc[0]),
                    "messages": messages})
    out.sort(key=lambda p: (p["nickname"], p["date"]))
    return out


def build_docx(corpora: list[str], *, negatives: int = 0,
               seed: int = 0) -> tuple[Document, int]:
    """The clinician script: every flagged phase, or -- with `negatives` --
    a random sample of that many screen-negative phases per corpus instead,
    in the identical layout so the same `collect` reads the filled copy."""
    import numpy as np

    rng = np.random.default_rng(seed)
    document = Document()
    document.styles["Normal"].font.name = FONT
    document.styles["Normal"].font.size = Pt(BODY_SIZE)
    document.styles["Normal"].paragraph_format.space_after = Pt(2)

    n_total = 0
    for corpus in corpora:
        path = output_path(corpus)
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} -- run "
                "`uv run python -m src.stage3.detect_loneliness --corpus "
                f"{corpus} --allow-network` first")
        scored = pd.read_parquet(path)
        dialogue = load_dialogue(corpus)

        phases = (negative_phases(corpus, dialogue, scored, negatives, rng)
                  if negatives else flagged_phases(corpus, dialogue, scored))
        for phase in phases:
            _paragraph(document, f"{phase['nickname']} — {phase['cid8']}",
                      bold=True, size=HEADER_SIZE)
            for message in phase["messages"]:
                _paragraph(document, f"{message['speaker']}: {message['text']}")

            checkbox_p = document.add_paragraph()
            checkbox_p.paragraph_format.space_after = Pt(2)
            label = checkbox_p.add_run("Expresses loneliness:  ")
            label.font.name, label.font.size, label.bold = FONT, Pt(BODY_SIZE), True
            _add_checkbox(checkbox_p, phase["phase_id"])

            explanation = _paragraph(document, EXPLANATION_LABEL, bold=True)
            explanation.paragraph_format.space_after = Pt(10)
            n_total += 1

    return document, n_total


# ================================================================ collecting


def _strip_prefix(text: str, prefix: str) -> str:
    text = text.strip()
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):].strip()
    return text


def _explanations_by_phase_id(document: Document) -> dict[str, str]:
    """The clinician's typed text under each phase's "Explanation:" line,
    keyed by that phase's phase_id.

    The phase_id comes from the checkbox paragraph immediately above (see
    _checkbox_tag_in_paragraph); the block ends at the next phase's bold
    header paragraph (see _is_header_paragraph), whatever the clinician did
    or didn't type in between.
    """
    explanations: dict[str, str] = {}
    current_tag = None
    collecting = False
    buf: list[str] = []

    def flush():
        if current_tag is not None:
            explanations[current_tag] = " ".join(s for s in buf if s).strip()

    for para in document.paragraphs:
        if _is_header_paragraph(para):
            flush()
            current_tag, collecting, buf = None, False, []
            continue
        tag = _checkbox_tag_in_paragraph(para)
        if tag is not None:
            current_tag = tag
            continue
        text = para.text.strip()
        if text.lower().startswith(EXPLANATION_LABEL.lower()):
            collecting = True
            buf = [_strip_prefix(text, EXPLANATION_LABEL)]
            continue
        if collecting:
            buf.append(text)
    flush()
    return explanations


def collect(docx_path) -> pd.DataFrame:
    document = Document(str(docx_path))
    found = dict(_iter_checkboxes(document))
    if not found:
        raise SystemExit(
            f"no checkboxes found in {docx_path} -- either nothing was "
            "flagged, or the file was re-saved by something that dropped "
            "the checkbox content controls (see the module docstring)")
    explanations = _explanations_by_phase_id(document)

    frames = []
    for corpus in CORPORA:
        path = output_path(corpus)
        if path.exists():
            frames.append(pd.read_parquet(path))
    scored = pd.concat(frames, ignore_index=True)
    flagged = scored[scored["lonely"] == 1].copy()

    missing_in_doc = set(flagged["phase_id"]) - set(found)
    unknown_in_doc = set(found) - set(flagged["phase_id"])
    if missing_in_doc:
        print(f"  WARNING: {len(missing_in_doc)} flagged phases have no "
              f"checkbox in the document -- they will be absent from "
              f"{LABELS_CSV.name}: {sorted(missing_in_doc)[:5]}"
              + (" ..." if len(missing_in_doc) > 5 else ""))
    if unknown_in_doc:
        print(f"  WARNING: {len(unknown_in_doc)} checkboxes in the document "
              "don't match any flagged phase in the screened parquet files "
              f"(stale corpus data?): {sorted(unknown_in_doc)[:5]}"
              + (" ..." if len(unknown_in_doc) > 5 else ""))

    flagged["clinician_lonely"] = [
        int(found[pid]) if pid in found else pd.NA for pid in flagged["phase_id"]]
    flagged["clinician_explanation"] = [
        explanations.get(pid, "") for pid in flagged["phase_id"]]
    out = flagged[["phase_id", "corpus", "nickname", "conversation_id", "type",
                   "date", "time", "lonely", "clinician_lonely",
                   "clinician_explanation"]].rename(columns={"lonely": "model_lonely"})
    return out


# ======================================================================= CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true",
                       help="write results/stage3/clinician_script.docx from "
                            "the already-screened phases")
    group.add_argument("--collect", metavar="DOCX",
                       help="read a filled-in copy of the document and write "
                            "results/stage3/clinician_labels.csv")
    ap.add_argument("--corpus", action="append", choices=CORPORA, default=None,
                    help="--build only; repeatable; defaults to all corpora")
    ap.add_argument("--negatives", type=int, default=0, metavar="N",
                    help="--build only: instead of the flagged phases, N "
                         "random screen-NEGATIVE phases per corpus, written "
                         "to clinician_script_negatives.docx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.build:
        corpora = args.corpus or list(CORPORA)
        document, n_total = build_docx(corpora, negatives=args.negatives,
                                       seed=args.seed)
        STAGE3.mkdir(parents=True, exist_ok=True)
        target = (DOCX_PATH.with_name("clinician_script_negatives.docx")
                  if args.negatives else DOCX_PATH)
        document.save(str(target))
        print(f"wrote {target.relative_to(ROOT)} -- {n_total} checkboxes "
              "across " + ", ".join(corpora))
        print("  open it in Microsoft Word yourself and confirm the boxes "
              "are checkable before sending it to a clinician")
        record_decision("stage3.clinician_docx.build", {
            "docx": str(DOCX_PATH.relative_to(ROOT)),
            "corpora": corpora,
            "checkboxes": n_total,
        })
    else:
        labels = collect(args.collect)
        STAGE3.mkdir(parents=True, exist_ok=True)
        labels.to_csv(LABELS_CSV, index=False)
        n_labeled = int(labels["clinician_lonely"].notna().sum())
        n_explained = int((labels["clinician_explanation"] != "").sum())
        print(f"wrote {LABELS_CSV.relative_to(ROOT)} -- {n_labeled} of "
              f"{len(labels)} flagged phases carry a clinician checkbox "
              f"state ({int(labels['clinician_lonely'].sum())} checked, "
              f"{n_explained} with a typed explanation)")
        record_decision("stage3.clinician_docx.collect", {
            "source_docx": str(args.collect),
            "labels_csv": str(LABELS_CSV.relative_to(ROOT)),
            "phases_flagged": int(len(labels)),
            "phases_labeled": n_labeled,
            "phases_checked": int(labels["clinician_lonely"].sum()),
            "phases_with_explanation": n_explained,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
