"""Stage 3, step 2: screen the conversations and write the clinician script.

How to run (sends participant speech to the Claude API; needs ANTHROPIC_API_KEY
in .env). Step 1 is `src.stage3.prepare_conversations`, which must have run:
    uv run python -m src.stage3.detect_loneliness --allow-network
    uv run python -m src.stage3.detect_loneliness --corpus deployment2 --allow-network
    uv run python -m src.stage3.detect_loneliness --limit 5 --allow-network   # smoke
    uv run python -m src.stage3.detect_loneliness --overwrite --allow-network # re-screen
    uv run python -m src.stage3.detect_loneliness --render-only               # rebuild the
                                                        # script from what is on disk, no API

Outputs (both under results/stage3/, so the screen and the document it feeds
sit together):
    results/stage3/screened_phases_{corpus}.parquet  one row per phase
    results/stage3/clinician_script.md               what a clinician reads

Input: data/conversation/{corpus}/dialogue.csv, the merged and cleaned table
`src.stage3.prepare_conversations` writes. Every line read here is
`content_clean` -- the verbatim `content_raw` beside it is used only where a
phase came back from the cleaning pass unusable, and a file that was never
cleaned at all says so on load rather than screening raw text silently.

This file is the whole of step 2: the screening call and the script it feeds
were two modules and one intermediate format, which meant three files had to
agree about what a turn was. They are one pass over one table now.

THE UNIT IS A CONVERSATION PHASE, NOT A TURN
--------------------------------------------
One request per (conversation_id, type) group -- the same grouping the
cleaning pass uses, and the same grouping the clinician reads. Both speakers'
messages go in, in order.

That follows from what these transcripts are. An older-adult turn here has a
median of 5 words: "Sometimes alone." is not judgeable on its own, and a
per-turn screen spends most of its prompt on the rubric and most of its
judgement on the template. A phase carries the exchange the reply sits in, so
the screen reads what a clinician would read.

The consequence for the outputs: a phase is 1 or 0, and no single turn inside
it is marked. The script therefore reproduces flagged phases whole and marks
nothing inside them, which is also what keeps the clinician's own reading
unanchored.

A MUCH SHORTER, LESS STRICT PROMPT THAN STAGE 1
------------------------------------------------
Stage 1 labels posts against `clinical_rubric`: four strict annotation rules
that keep a benchmark's positives clean, requiring an explicit relational
appraisal and excluding conflict and distress.

Stage 3 is not labelling anything. It decides which conversations a
psychiatrist reads, and the clinician's own reading is the ground truth that
comes out of it. So it carries `screening_prompt` (configs/prompts.yaml,
SHA-256 pinned like every other construct text): the construct's definition
sentence, the screening role, and one binary question. No annotation rules at
all -- none of Stage 1's four exclusions apply here, and the screen is
expected to be more permissive than the annotation standard.

Stated plainly: Stage 3's flag rate is not comparable to Stage 1's positive
rate, and no Stage 3 number is a detection result. What the clinician marks on
this script is the result.

WHY CLAUDE, AND NOT A STUDY ARM
-------------------------------
The screen selects which conversations become out-of-domain ground truth.
Selecting them with one of the arms under evaluation and then scoring that arm
on what it chose would bias the evaluation toward it. Claude is outside the FT
arms and the probe family, consumes no LonXplain label, inherits no fold and
fits no threshold.

PHI
---
Participant speech goes to a third-party API. That happens only under
--allow-network, and the Batch API retains results for 29 days.
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from src.baselines.prompting import (SCREENING_BLOCK_SHA256, SCREENING_PROMPT,
                                     SCREENING_PROMPT_SHA256,
                                     build_screening_prompt,
                                     check_construct_hashes)
from src.stage3.prepare_conversations import CORPORA, corpus_dir
from src.utils.common import ROOT, STAGE3, record_decision

SCRIPT = STAGE3 / "clinician_script.md"

# A phase whose older-adult messages total fewer than this many words is
# acknowledgement, not evidence, and is not sent. It is still written to the
# parquet, flagged 0 and marked `screened=False`, so the table accounts for
# every phase in the corpus rather than quietly covering some of them.
MIN_PHASE_USER_WORDS = 4

CLAUDE_MODEL = "claude-opus-5"
CLAUDE_EFFORT = "low"      # one bounded binary judgement, no contract to fill
MAX_TOKENS = 512           # the answer is one digit; this is headroom, not need

USER = "OLDER ADULT"
BOT = "CHATBOT"

PHASE_COLUMNS = ["corpus", "phase_id", "conversation_id", "type", "nickname",
                 "date", "time", "n_messages", "n_user_messages",
                 "n_user_words", "screened", "model", "lonely", "answer"]


# =========================================================== candidate phases


def load_dialogue(corpus: str) -> pd.DataFrame:
    """The cleaned, merged dialogue for one corpus."""
    path = corpus_dir(corpus) / "dialogue.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} — run "
            "`uv run python -m src.stage3.prepare_conversations "
            "--allow-network` first")
    dialogue = pd.read_csv(path).fillna({"content_clean": "", "content_raw": ""})
    # Fraction, not `.any()`. A file with one cleaned phase out of 519 is a
    # raw file, and an `.any()` check called it clean -- which is exactly how
    # a screening run once went out on 99.2% raw text without saying so.
    n_clean = int(dialogue["cleaned"].sum())
    if n_clean < len(dialogue):
        print(f"  WARNING: {corpus} dialogue.csv is cleaned for {n_clean} of "
              f"{len(dialogue)} messages ({100 * n_clean / len(dialogue):.1f}%)"
              "; the rest is screened as merged raw text. Finish it with "
              "`uv run python -m src.stage3.prepare_conversations --corpus "
              f"{corpus} --allow-network`")
    return dialogue


def message_text(row) -> str:
    """One message's text: cleaned where the cleaning pass reached it."""
    return str(row.content_clean).strip() or str(row.content_raw).strip()


def exchange_text(rows: pd.DataFrame) -> str:
    """One phase rendered for the prompt: both speakers, in order."""
    return "\n".join(f"{row.speaker}: {message_text(row)}"
                     for row in rows.sort_values("order",
                                                 kind="stable").itertuples()
                     if message_text(row))


def candidate_phases(dialogue: pd.DataFrame) -> pd.DataFrame:
    """One row per (conversation_id, type) phase, with its exchange text."""
    rows = []
    for (cid, phase), group in dialogue.groupby(["conversation_id", "type"],
                                                sort=False):
        group = group.sort_values("order", kind="stable")
        user = group[group["speaker"] == USER]
        user_words = int(user["n_words"].sum())
        rows.append({
            "corpus": str(group["corpus"].iloc[0]),
            "phase_id": f"{group['corpus'].iloc[0]}_{cid[:8]}_{phase}",
            "conversation_id": cid,
            "type": str(phase),
            "nickname": str(group["nickname"].iloc[0]),
            "date": str(group["date"].iloc[0]),
            "time": str(group["time"].iloc[0]),
            "n_messages": int(len(group)),
            "n_user_messages": int(len(user)),
            "n_user_words": user_words,
            "screened": user_words >= MIN_PHASE_USER_WORDS,
            "exchange": exchange_text(group),
        })
    phases = pd.DataFrame(rows)
    n_send = int(phases["screened"].sum())
    print(f"  {n_send} of {len(phases)} phases reach the "
          f"{MIN_PHASE_USER_WORDS}-word floor and will be screened", flush=True)
    return phases


# ================================================================= the screen


ANSWER = re.compile(r"[01]")


def parse_answer(text: str) -> int | None:
    """The single digit, or None when the response is not one.

    A bare "1"/"0" is the contract. Anything else is read only if it contains
    exactly one of the two digits; a response carrying both is ambiguous and
    rejected rather than resolved by position.
    """
    digits = ANSWER.findall(text.strip())
    if len(digits) != 1:
        return None
    return int(digits[0])


def screen(phases: pd.DataFrame, *, poll_seconds: int = 30) -> pd.DataFrame:
    """1/0 per phase from Claude Opus 5, through the Batch API.

    Batch rather than live requests: there is no latency requirement and batch
    is half price. A phase whose answer does not parse is recorded 0 with its
    raw text kept in `answer`, and counted -- never guessed at.
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from src.utils.common import load_dotenv

    sendable = phases[phases["screened"]]
    load_dotenv()
    client = anthropic.Anthropic()
    requests = [
        Request(
            custom_id=f"phase-{i}",
            params=MessageCreateParamsNonStreaming(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": CLAUDE_EFFORT},
                messages=[{"role": "user",
                           "content": build_screening_prompt(row.exchange)}],
            ),
        )
        for i, row in zip(sendable.index, sendable.itertuples())
    ]

    print(f"  screening: submitting {len(requests)} conversation phases to the "
          f"Batch API ({CLAUDE_MODEL}, effort={CLAUDE_EFFORT})", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"  screening: batch {batch.id}", flush=True)
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"    {batch.processing_status}: {counts.succeeded} done, "
              f"{counts.processing} processing, {counts.errored} errored",
              flush=True)
        time.sleep(poll_seconds)

    answers, labels, unusable = {}, {}, 0
    for result in client.messages.batches.results(batch.id):
        i = int(result.custom_id.removeprefix("phase-"))
        if result.result.type != "succeeded":
            answers[i] = f"<{result.result.type}>"
            unusable += 1
            continue
        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), "")
        answers[i] = text.strip()[:200]
        label = parse_answer(text)
        if label is None:
            unusable += 1
        else:
            labels[i] = label
    if unusable:
        print(f"  screening: {unusable} of {len(requests)} phases returned no "
              "usable digit; recorded 0 with the raw answer kept", flush=True)

    out = phases.copy()
    out["model"] = CLAUDE_MODEL
    out["lonely"] = [int(labels.get(i, 0)) for i in out.index]
    out["answer"] = [answers.get(i, "") for i in out.index]
    return out[PHASE_COLUMNS]


def output_path(corpus: str):
    return STAGE3 / f"screened_phases_{corpus}.parquet"


# ====================================================== the clinician script


def render(corpus_blocks: list[dict]) -> str:
    """The markdown a clinician reads: conversation first, phase second."""
    n_conv = sum(len(b["conversations"]) for b in corpus_blocks)
    n_phase = sum(len(c["phases"]) for b in corpus_blocks
                  for c in b["conversations"])
    people = {c["nickname"] for b in corpus_blocks for c in b["conversations"]}
    return "\n".join([
        "# Loneliness script for clinical review",
        "",
        f"{n_phase} conversation phases from {n_conv} conversations and "
        f"{len(people)} participants, included because a screen judged that "
        "the older adult expressed loneliness in them.",
        "",
        "**How this was built.** Each conversation phase — one conversation, "
        "one phase type, both speakers, in order — was read whole by Claude "
        "Opus 5 against the screening prompt below, which answers 1 or 0. The "
        "model was trained on none of this study's corpora and fits no "
        "threshold. Phases it answered 1 for are reproduced here in full, "
        "chatbot turns included.",
        "",
        "**This is a screen, not a diagnosis.** The prompt carries the "
        "definition of the construct and no annotation rules, so it is "
        "deliberately more permissive than the rubric the study's written "
        "corpus was labelled against. Expect phases you disagree with — that "
        "is the screen doing its job rather than failing at it.",
        "",
        "**What to check.** For each phase: does the older adult express "
        "loneliness as you would judge it clinically? Nothing inside a phase "
        "is marked. The screen judged the exchange as a whole, and showing "
        "you where it thought the evidence was would anchor the judgement "
        "this document exists to collect.",
        "",
        "**Transcript quality.** These are speech-to-text transcripts of "
        "spoken conversation. Consecutive fragments from one speaker have "
        "been joined into one message, and filler sounds, stutters and "
        "recogniser duplications were removed by a cleaning pass that was "
        "forbidden to add or reword anything. The verbatim text of every line "
        "is kept in `data/conversation/{corpus}/dialogue.csv`.",
        "",
        "---",
        "",
        "## The screening prompt the model applied",
        "",
        "\n".join("> " + line if line else ">"
                  for line in SCREENING_PROMPT.strip().splitlines()),
        "",
        "---",
        "",
        *_body(corpus_blocks),
    ]) + "\n"


def _body(corpus_blocks: list[dict]) -> list[str]:
    lines = []
    for block in corpus_blocks:
        lines += [f"# Corpus: {block['corpus']}", ""]
        for conversation in block["conversations"]:
            lines += [
                f"## {conversation['nickname']} — conversation "
                f"`{conversation['cid'][:8]}` ({conversation['date']})",
                "",
                f"{len(conversation['phases'])} phase(s) flagged of "
                f"{conversation['n_phases']} in this conversation.",
                "",
            ]
            for phase in conversation["phases"]:
                lines += [
                    f"### {phase['type']} · {phase['n_user']} older-adult "
                    f"turns · from {phase['time']}",
                    "",
                    "```",
                    *(f"{message['time']}  {message['speaker']:<11}: "
                      f"{message['text']}" for message in phase["messages"]),
                    "```",
                    "",
                ]
    return lines


def build_blocks(corpus: str, dialogue: pd.DataFrame,
                 scored: pd.DataFrame) -> dict:
    """One corpus's flagged phases, grouped conversation first."""
    flagged = {(row.conversation_id, row.type)
               for row in scored.itertuples() if int(row.lonely) == 1}
    per_conversation = scored.groupby("conversation_id").size().to_dict()

    conversations = []
    wanted = {cid for cid, _ in flagged}
    scoped = dialogue[dialogue["conversation_id"].isin(wanted)]
    for cid, group in scoped.groupby("conversation_id", sort=False):
        phases = []
        for phase, phase_rows in group.groupby("type", sort=False):
            if (cid, phase) not in flagged:
                continue          # an unflagged phase is not reproduced
            rows = phase_rows.sort_values("order", kind="stable")
            messages = [{"time": row.time, "speaker": row.speaker,
                         "text": message_text(row)}
                        for row in rows.itertuples() if message_text(row)]
            phases.append({
                "type": str(phase),
                "time": str(rows["time"].iloc[0]),
                "messages": messages,
                "n_user": sum(m["speaker"] == USER for m in messages),
            })
        if not phases:
            continue
        conversations.append({
            "cid": cid,
            "nickname": str(group["nickname"].iloc[0]),
            "date": str(group["date"].min()),
            "phases": phases,
            "n_phases": int(per_conversation.get(cid, len(phases))),
        })
    conversations.sort(key=lambda c: (c["nickname"], c["date"]))
    return {"corpus": corpus, "conversations": conversations}


# ======================================================================= CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", choices=CORPORA, default=None,
                    help="repeatable; defaults to both conversation corpora")
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N phases per corpus (a dry "
                         "run; written to a _limit{N} stem so a partial run "
                         "can never be mistaken for a full one)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-screen corpora whose parquet already exists")
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild results/clinician_script.md from the "
                         "parquet files already on disk; makes no API call")
    ap.add_argument("--allow-network", action="store_true",
                    help="explicit approval to send participant speech to the "
                         "Claude API")
    args = ap.parse_args()
    if not (args.allow_network or args.render_only):
        ap.error("screening sends participant speech to a third-party API; "
                 "pass --allow-network to approve it, or --render-only to "
                 "rebuild the script from what is already screened")
    check_construct_hashes()

    corpora = args.corpus or list(CORPORA)
    STAGE3.mkdir(parents=True, exist_ok=True)

    blocks, totals = [], {"screened": 0, "flagged": 0, "conversations": 0}
    for corpus in corpora:
        suffix = "" if args.limit is None else f"_limit{args.limit}"
        path = output_path(corpus + suffix)
        dialogue = load_dialogue(corpus)

        if args.render_only or (path.exists() and not args.overwrite):
            if not path.exists():
                print(f"{corpus}: {path.name} does not exist yet — screen it "
                      "with --allow-network first")
                continue
            scored = pd.read_parquet(path)
            print(f"{corpus}: reusing {path.name} "
                  f"({int(scored['lonely'].sum())} flagged of {len(scored)})")
        else:
            print(f"{corpus}:", flush=True)
            cleaned_fraction = float(dialogue["cleaned"].mean())
            phases = candidate_phases(dialogue)
            if args.limit is not None:
                phases = phases.head(args.limit).copy()
                print(f"  DRY RUN: {len(phases)} phases", flush=True)
            scored = screen(phases)
            scored.to_parquet(path, index=False)
            n_flag = int(scored["lonely"].sum())
            sent = int(scored["screened"].sum())
            print(f"  {n_flag} of {sent} screened phases flagged "
                  f"({100 * n_flag / max(sent, 1):.1f}%); wrote "
                  f"{path.relative_to(ROOT)}", flush=True)
            if args.limit is None:
                record_decision(f"stage3.screen.{corpus}", {
                    "model": CLAUDE_MODEL,
                    "effort": CLAUDE_EFFORT,
                    "unit": "one (conversation_id, type) phase, both speakers",
                    "prompt": "screening_prompt (definition sentence + role, "
                              "NOT the Stage 1 clinical rubric)",
                    "screening_prompt_sha256": SCREENING_PROMPT_SHA256,
                    "screening_block_sha256": SCREENING_BLOCK_SHA256,
                    "min_phase_user_words": MIN_PHASE_USER_WORDS,
                    # What was actually screened: a run over a half-cleaned
                    # dialogue is a different run, and the record says which.
                    "dialogue_cleaned_fraction": round(cleaned_fraction, 4),
                    "decision_rule": "the model's own 1/0; no threshold "
                                     "fitted on any label",
                    "phases_total": int(len(scored)),
                    "phases_screened": sent,
                    "phases_flagged": n_flag,
                })

        block = build_blocks(corpus, dialogue, scored)
        blocks.append(block)
        totals["screened"] += int(scored["screened"].sum())
        totals["flagged"] += int(scored["lonely"].sum())
        totals["conversations"] += len(block["conversations"])

    if not blocks:
        print("nothing to render")
        return 1

    SCRIPT.write_text(render(blocks))
    people = {c["nickname"] for b in blocks for c in b["conversations"]}
    print(f"\nwrote {SCRIPT.relative_to(ROOT)} — {totals['flagged']} flagged "
          f"phases across {totals['conversations']} conversations and "
          f"{len(people)} participants, of {totals['screened']} phases "
          "screened")
    record_decision("stage3.clinician_script", {
        "script": str(SCRIPT.relative_to(ROOT)),
        "inclusion": "every phase the screen answered 1 for, reproduced whole "
                     "with chatbot turns; nothing marked inside it",
        "grouping": "conversation_id, then phase type",
        "phases_screened": totals["screened"],
        "phases_flagged": totals["flagged"],
        "conversations": totals["conversations"],
        "participants": len(people),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
