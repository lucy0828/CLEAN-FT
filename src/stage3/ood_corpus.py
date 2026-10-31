"""The out-of-domain corpus as the detectors and the annotators both see it.

Not run directly. `src.stage3.ood_detection` scores the phases this module
returns; `src.stage3.annotator_labels` aligns the clinicians' quoted spans to
the same text. Both have to agree, character for character, on what a phase's
text IS -- a span located here is later scored against word attributions
computed on the identical string -- so it is built in exactly one place.

THE UNIT IS THE PHASE, AND THE TEXT IS THE OLDER ADULT'S ALONE
--------------------------------------------------------------
One row per (conversation_id, type) phase -- the unit the step-2 screen
judged and the clinician script reproduces. The text is the older adult's
merged messages in that phase, in order, joined by newlines: a Stage 1
detector was trained on one person's own words, and feeding it the chatbot's
questions would add a second, uncontrolled distribution shift on top of the
one under study. The screen and the clinician read the whole exchange and the
detectors read half of it; `scored_speaker` is written on every raw row so the
asymmetry is on the record rather than in a docstring.

Only phases the screen actually read are returned. A phase below
`detect_loneliness.MIN_PHASE_USER_WORDS` was recorded 0 without being sent, so
it is not part of the corpus the clinician script was drawn from.
"""

from __future__ import annotations

import pandas as pd

from src.stage3.detect_loneliness import USER, load_dialogue, message_text, output_path
from src.stage3.prepare_conversations import CORPORA
from src.utils.common import ROOT

SPEAKER = USER


def load_phases(corpora=CORPORA) -> tuple[pd.DataFrame, dict]:
    """The screened phases, the older adult's words in each, and turn bounds.

    Returns (phases, turns_of). `phases` carries one row per screened phase
    with its `text`; `turns_of` maps phase_id -> [(char_start, char_end,
    order)], one entry per older-adult message, so a word can be attributed
    to the turn it sits in rather than only to the phase.
    """
    frames, turns_of = [], {}
    for corpus in corpora:
        path = output_path(corpus)
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} -- screen the corpus first "
                "with `uv run python -m src.stage3.detect_loneliness "
                "--allow-network`")
        screened = pd.read_parquet(path)
        screened = screened[screened["screened"].astype(bool)]
        dialogue = load_dialogue(corpus)

        speech = {}
        user = dialogue[dialogue["speaker"] == SPEAKER]
        for (cid, phase), group in user.groupby(["conversation_id", "type"],
                                                sort=False):
            parts, bounds, cursor = [], [], 0
            for row in group.sort_values("order", kind="stable").itertuples():
                line = message_text(row)
                if not line:
                    continue
                bounds.append((cursor, cursor + len(line), int(row.order)))
                parts.append(line)
                cursor += len(line) + 1          # the "\n" the join inserts
            speech[(cid, phase)] = ("\n".join(parts), bounds)

        rows = []
        for row in screened.itertuples():
            text, bounds = speech.get((row.conversation_id, row.type), ("", []))
            if not text.strip():
                raise SystemExit(
                    f"{row.phase_id} was screened but carries no {SPEAKER} "
                    "text in dialogue.csv")
            turns_of[row.phase_id] = bounds
            rows.append({
                "corpus": row.corpus, "phase_id": row.phase_id,
                "conversation_id": row.conversation_id, "type": row.type,
                "nickname": row.nickname, "date": row.date,
                "n_user_messages": int(row.n_user_messages),
                "n_user_words": int(row.n_user_words),
                "in_clinician_script": bool(int(row.lonely) == 1),
                "text": text,
            })
        frames.append(pd.DataFrame(rows))

    phases = pd.concat(frames, ignore_index=True)
    assert phases["phase_id"].is_unique, "duplicate phase_id across corpora"
    return phases, turns_of
