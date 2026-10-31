"""Stage 3, step 1: build and clean the older-adult conversation corpora.

How to run (the cleaning pass sends participant speech to the Claude API, so
it needs ANTHROPIC_API_KEY in .env and explicit approval):
    uv run python -m src.stage3.prepare_conversations --allow-network
    uv run python -m src.stage3.prepare_conversations --corpus deployment2 --allow-network
    uv run python -m src.stage3.prepare_conversations --limit-groups 3 --allow-network   # smoke
    uv run python -m src.stage3.prepare_conversations                  # tables only, no cleaning
    uv run python -m src.stage3.prepare_conversations --overwrite --allow-network  # re-clean

Step 2 is `src.stage3.detect_loneliness`, which screens the cleaned dialogue
and writes the clinician script.

Inputs:
    data/conversation/deployment1.csv
    data/conversation/deployment2.csv
    data/conversation/deployment2b.csv
        columns: nickname, conversation_id, timestamp, source, type, content
        deployment2b is a later incremental export of the same corpus as deployment2
        and can share whole conversation_ids with it (the newest conversation
        as of deployment2's cutoff was re-exported byte-identical in deployment2b).
        Those conversation_ids are dropped from deployment2b before anything else
        runs -- see drop_cross_corpus_overlap -- so no conversation is
        screened, or could appear in the clinician script, twice.

Outputs (per corpus, under data/conversation/{corpus}/, so the raw export and
everything built from it stay together):
    turns.csv    — one row per USER turn: id, nickname, conversation_id,
                   timestamp, type, content, n_words, corpus
    sessions.csv — one row per session: conversation_id, nickname,
                   n_user_turns, date
    dropped_duplicate_rows.csv — audit of every discarded duplicate row
    dialogue.csv — the READABLE conversation: one row per merged message,
                   grouped by conversation_id and then by phase `type`, with
                   nickname, date, time (HH:mm:ss), speaker, the raw text and
                   the cleaned text side by side

THE CLEANING PASS (dialogue.csv)
--------------------------------
turns.csv is the analysis table and stays verbatim. dialogue.csv is what a
labeller reads, and these transcripts are speech-to-text either side: the
older adult's speech arrives with filler sounds and recogniser stutters, and
the chatbot's TTS text arrives split mid-sentence. Two repairs, in this order:

  1. RUN MERGING (local, lossless). Consecutive messages from the same speaker
     inside one (conversation_id, type) phase become one message, joined in
     timestamp order. A reply the recogniser cut into six fragments is read as
     one utterance. The merged row keeps the first fragment's timestamp and
     records how many rows it came from, so nothing is hidden.
  2. FILLER REMOVAL (Claude Opus 5, Batch API). Each phase is sent as one
     request under the SHA-256-pinned cleanup prompt, which removes filler
     sounds, stutters and duplicated words and is forbidden to add, complete
     or reword anything. Both texts are kept: `content_raw` is what the
     recogniser produced, `content_clean` is what the clinician reads. A phase
     whose response does not come back with one cleaned text per message keeps
     its raw text and is counted, never guessed at.

This deliberately reverses an earlier choice to leave disfluency in as
clinical signal. The signal is not lost -- `content_raw` carries it, and
`n_merged` records the fragmentation -- but the document a labeller reads is
now the cleaned one.

Rules:
    - Only source=user rows become turns. Assistant text is never a row.
    - Turn id = "{corpus}_{conversation_id[:8]}_{row_index}", where row_index
      is the row's 0-based position in the raw CSV, so the id is stable no
      matter how the table is later sorted and the corpus prefix stops
      deployment1 and deployment2 ids from colliding.
    - Both raw CSVs log rows twice (all six columns identical, same
      microsecond timestamp): 717 rows in deployment1 and 1,304 in deployment2, of
      which 476 and 829 are user turns. These are dropped by
      `drop_exact_duplicates` before anything else runs, with an audit CSV of
      every discarded row. The duplication rate varies from 0%% to 36%% BY
      PARTICIPANT, so leaving it in biases any per-participant quantity —
      this is a data-integrity repair, not a preprocessing preference.

Sanity checks (the module fails loudly if violated): post-deduplication turn /
user-turn / session / participant counts match EXPECTED, and turn ids are
unique.
"""

import argparse
import json
import sys
import time

import pandas as pd

from src.utils.common import record_decision
from src.utils.data_prep import DATA
from src.utils.data_prep import words

CORPORA = ("deployment1", "deployment2", "deployment2b")

# Source CSVs and everything derived from them live together under
# data/conversation/, one subdirectory per corpus, so the raw export and the
# files built from it are never separated.
CONVERSATION = DATA / "conversation"


def corpus_dir(corpus: str):
    """Where one corpus's derived files are written."""
    return CONVERSATION / corpus


# Checked on every run. These are the counts AFTER exact-duplicate removal
# (see drop_exact_duplicates) and, for deployment2b, after the cross-corpus overlap
# removal (see drop_cross_corpus_overlap) too. The raw export counted
# 717 + 1,304 duplicated rows as speech; session and participant counts are
# unchanged, because no session consists only of duplicates. deployment2b's raw
# export additionally repeats one whole conversation (31 rows) already in
# deployment2.csv, dropped before the exact-duplicate count below is taken.
EXPECTED = {
    "deployment1": {"rows": 8244, "user_turns": 5520, "sessions": 192, "participants": 10},
    "deployment2": {"rows": 8668, "user_turns": 5231, "sessions": 181, "participants": 8},
    "deployment2b": {"rows": 408, "user_turns": 239, "sessions": 10, "participants": 3},
}

TURN_COLUMNS = ["id", "nickname", "conversation_id", "timestamp", "type",
                "content", "n_words", "corpus"]
SESSION_COLUMNS = ["conversation_id", "nickname", "n_user_turns", "date"]


def n_words(text: str) -> int:
    """Word count under the study's single word tokenizer (src.utils.data_prep)."""
    return len(words(text))


def drop_exact_duplicates(raw: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Drop rows identical in all six raw columns, including the timestamp.

    A participant cannot utter the same sentence twice at the same microsecond,
    so a byte-identical row with a byte-identical microsecond timestamp is a
    logging defect, never speech. Removing it deletes no word anyone said —
    this is a data-integrity repair, not a preprocessing choice.

    It is also not cosmetic: the per-participant duplication rate runs from
    0%% to 36%% and correlates r = +0.63 with a participant's clinical
    activation per token. Left in, the defect reads as a clinical finding.

    The counts dropped here are written to results/decisions.json, and an audit
    CSV of every discarded row is written beside the corpus so the deletion is
    inspectable rather than asserted.
    """
    before = len(raw)
    duplicated = raw.duplicated(subset=RAW_COLUMNS, keep="first")
    dropped = raw[duplicated]
    kept = raw[~duplicated].copy()

    audit_dir = corpus_dir(corpus)
    audit_dir.mkdir(parents=True, exist_ok=True)
    dropped[["row_index"] + RAW_COLUMNS].to_csv(
        audit_dir / "dropped_duplicate_rows.csv", index=False
    )
    by_source = dropped["source"].value_counts().to_dict()
    print(f"  {corpus}: dropped {before - len(kept)} exact-duplicate rows "
          f"({by_source}) -> {len(kept)} rows; "
          f"audit at {(audit_dir / 'dropped_duplicate_rows.csv').relative_to(DATA.parent)}")
    return kept


RAW_COLUMNS = ["nickname", "conversation_id", "timestamp", "source", "type",
               "content"]

# deployment2b is an incremental export pulled after deployment2.csv, from the same
# corpus. The two files can share whole conversation_ids -- the newest
# conversation as of deployment2's cutoff was re-exported byte-identical in
# deployment2b -- so those conversations are dropped from the newer file. Left in,
# one clinical conversation would be screened, and could be published in the
# clinician script, under two different corpus headings.
OVERLAP_WITH = {"deployment2b": "deployment2"}


def drop_cross_corpus_overlap(raw: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Drop rows whose conversation_id also appears in an earlier corpus export.

    See OVERLAP_WITH. A no-op for any corpus not listed there. The dropped
    rows are written to an audit CSV beside the corpus, same as
    drop_exact_duplicates, so the repair is inspectable rather than asserted.
    """
    other = OVERLAP_WITH.get(corpus)
    if other is None:
        return raw
    other_cids = set(
        pd.read_csv(CONVERSATION / f"{other}.csv",
                    usecols=["conversation_id"])["conversation_id"])
    overlap = raw["conversation_id"].isin(other_cids)
    dropped = raw[overlap]
    kept = raw[~overlap].copy()
    if len(dropped):
        audit_dir = corpus_dir(corpus)
        audit_dir.mkdir(parents=True, exist_ok=True)
        dropped[["row_index"] + RAW_COLUMNS].to_csv(
            audit_dir / "dropped_overlap_conversations.csv", index=False)
        print(f"  {corpus}: dropped {len(dropped)} rows from "
              f"{dropped['conversation_id'].nunique()} conversation(s) also "
              f"present in {other} -> {len(kept)} rows; audit at "
              f"{(audit_dir / 'dropped_overlap_conversations.csv').relative_to(DATA.parent)}")
    return kept


def load_raw(corpus: str) -> pd.DataFrame:
    """Read the raw conversation CSV, de-duplicated, keeping the row position."""
    path = CONVERSATION / f"{corpus}.csv"
    raw = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    assert not missing, f"{path} is missing columns {sorted(missing)}"
    # row_index is assigned BEFORE the drop so surviving rows keep the position
    # they had in the raw file, and therefore keep their turn id.
    raw = raw.reset_index(names="row_index")
    raw = drop_cross_corpus_overlap(raw, corpus)
    raw = drop_exact_duplicates(raw, corpus)
    # Parsed only for ordering; turns.csv keeps the original timestamp string.
    raw["_ts"] = pd.to_datetime(raw["timestamp"], format="mixed", utc=True)
    assert raw["_ts"].notna().all(), f"{path}: unparseable timestamps"
    return raw


def build_turns(raw: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """One row per user turn, in session-walk order."""
    user = raw[raw["source"] == "user"].copy()
    assert user["content"].notna().all(), f"{corpus}: user turn with empty content"
    user["content"] = user["content"].astype(str)

    # Sessions in first-appearance order; turns inside a session in timestamp
    # order. The sort is stable, so turns sharing a timestamp (13 sessions in
    # deployment1, 17 in deployment2) keep their raw file order.
    user["_session_order"] = user.groupby("conversation_id", sort=False).ngroup()
    user = user.sort_values(["_session_order", "_ts", "row_index"], kind="stable")

    turns = pd.DataFrame(
        {
            "id": [
                f"{corpus}_{cid[:8]}_{i}"
                for cid, i in zip(user["conversation_id"], user["row_index"])
            ],
            "nickname": user["nickname"].to_numpy(),
            "conversation_id": user["conversation_id"].to_numpy(),
            "timestamp": user["timestamp"].to_numpy(),
            "type": user["type"].to_numpy(),
            "content": user["content"].to_numpy(),
            "n_words": [n_words(c) for c in user["content"]],
            "corpus": corpus,
        }
    )
    assert turns["id"].is_unique, f"{corpus}: duplicate turn ids"
    return turns[TURN_COLUMNS]


def build_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per session, including sessions with no user turn."""
    rows = []
    for cid, session in raw.groupby("conversation_id", sort=False):
        nicknames = session["nickname"].unique()
        assert len(nicknames) == 1, f"session {cid} spans {len(nicknames)} nicknames"
        rows.append(
            {
                "conversation_id": cid,
                "nickname": nicknames[0],
                "n_user_turns": int((session["source"] == "user").sum()),
                # Sessions run minutes, not days; the date is the first turn's
                # UTC date (2 deployment1 and 1 deployment2 session cross midnight).
                "date": session["_ts"].min().date().isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


# ===================================================== the readable dialogue

DIALOGUE_COLUMNS = ["corpus", "conversation_id", "type", "nickname", "date",
                    "time", "order", "speaker", "content_clean", "content_raw",
                    "n_merged", "n_words", "row_indexes", "cleaned"]

# Both sides are machine speech pipelines, so both sides fragment: the older
# adult's STT splits one utterance across rows, the chatbot's TTS text arrives
# cut mid-sentence. Merging is applied to both, so an exchange reads as an
# exchange.
SPEAKER = {"user": "OLDER ADULT", "assistant": "CHATBOT"}

CLEANUP_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"},
                               "text": {"type": "string"}},
                "required": ["index", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["messages"],
    "additionalProperties": False,
}


def merge_runs(raw: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """One row per merged message, grouped by conversation and then by phase.

    Consecutive rows from the same speaker inside one (conversation_id, type)
    phase are one utterance that the speech pipeline cut up; they are joined in
    timestamp order. `row_indexes` records exactly which raw rows went into a
    merged message, so the merge is reversible from the file itself.
    """
    # `_ts` is renamed for the walk below: itertuples() mangles any column
    # whose name starts with an underscore into a positional field.
    ordered = raw.assign(ts=raw["_ts"]).sort_values(
        ["conversation_id", "type", "ts", "row_index"], kind="stable")
    rows = []
    for (cid, phase), group in ordered.groupby(["conversation_id", "type"],
                                               sort=False):
        current = None
        for row in group.itertuples():
            content = str(row.content).strip()
            if current is not None and current["speaker_source"] == row.source:
                current["parts"].append(content)
                current["row_indexes"].append(int(row.row_index))
                continue
            if current is not None:
                rows.append(current)
            current = {
                "corpus": corpus,
                "conversation_id": cid,
                "type": str(phase),
                "nickname": row.nickname,
                # Sessions run minutes; the date is the phase's own local date
                # and the time is the first fragment's, to the second.
                "date": row.ts.date().isoformat(),
                "time": row.ts.strftime("%H:%M:%S"),
                "speaker_source": row.source,
                "parts": [content],
                "row_indexes": [int(row.row_index)],
            }
        if current is not None:
            rows.append(current)

    out = []
    counters = {}
    for row in rows:
        key = (row["conversation_id"], row["type"])
        counters[key] = counters.get(key, 0) + 1
        text = " ".join(part for part in row["parts"] if part)
        out.append({
            "corpus": row["corpus"],
            "conversation_id": row["conversation_id"],
            "type": row["type"],
            "nickname": row["nickname"],
            "date": row["date"],
            "time": row["time"],
            "order": counters[key],
            "speaker": SPEAKER.get(row["speaker_source"], row["speaker_source"]),
            "content_clean": text,          # replaced by the cleaning pass
            "content_raw": text,
            "n_merged": len(row["row_indexes"]),
            "n_words": n_words(text),
            "row_indexes": json.dumps(row["row_indexes"]),
            "cleaned": False,
        })
    return pd.DataFrame(out, columns=DIALOGUE_COLUMNS)


def uncleaned_phases(dialogue: pd.DataFrame) -> list[tuple[str, str]]:
    """The (conversation_id, type) phases not every message of which is clean."""
    done = dialogue.groupby(["conversation_id", "type"])["cleaned"].all()
    return [key for key, complete in done.items() if not complete]


def clean_with_claude(dialogue: pd.DataFrame, *, limit_groups: int | None = None,
                      poll_seconds: int = 30) -> tuple[pd.DataFrame, dict]:
    """Filler removal, one Batch API request per (conversation, phase).

    A phase is sent whole rather than message by message: the model needs the
    surrounding turns to tell a recogniser's duplicated word from a speaker's
    deliberate repetition, and one request per phase is ~870 requests for the
    whole study rather than ~9,000.

    Only phases that are not already fully clean are sent, so an interrupted
    or partial pass RESUMES rather than re-spending the API on work already
    done -- and, more importantly, so a second run actually finishes the job
    instead of quietly leaving the rest raw.

    A response that does not return one cleaned text per message leaves that
    phase raw. The count of those is returned, not swallowed: a cleaning pass
    that silently half-ran would be indistinguishable from one that worked.
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from src.baselines.prompting import (CLEANUP_BLOCK_SHA256,
                                         build_cleanup_prompt,
                                         extract_json_object)
    from src.utils.common import load_dotenv

    todo = set(uncleaned_phases(dialogue))
    total = dialogue.groupby(["conversation_id", "type"]).ngroups
    groups = [(key, group) for key, group
              in dialogue.groupby(["conversation_id", "type"], sort=False)
              if key in todo]
    already = total - len(groups)
    if already:
        print(f"  {already} of {total} phases are already clean; sending the "
              f"remaining {len(groups)}")
    if limit_groups is not None:
        groups = groups[:limit_groups]
        print(f"  SMOKE RUN: cleaning {len(groups)} of {total} phases")
    if not groups:
        print("  every phase is already clean; nothing to send")
        return dialogue, {"cleanup_model": CLEANUP_MODEL, "phases_sent": 0,
                          "phases_unusable_left_raw": 0,
                          "messages_cleaned": int(dialogue["cleaned"].sum()),
                          "messages_changed": int(
                              (dialogue["content_clean"]
                               != dialogue["content_raw"]).sum()),
                          "messages_emptied": int(
                              (dialogue["content_clean"].str.strip() == "").sum())}

    load_dotenv()
    client = anthropic.Anthropic()
    requests, index_of = [], {}
    for n, (_key, group) in enumerate(groups):
        index_of[n] = group.index.to_numpy()
        requests.append(Request(
            custom_id=f"phase-{n}",
            params=MessageCreateParamsNonStreaming(
                model=CLEANUP_MODEL,
                max_tokens=CLEANUP_MAX_TOKENS,
                output_config={
                    "effort": CLEANUP_EFFORT,
                    "format": {"type": "json_schema", "schema": CLEANUP_SCHEMA},
                },
                messages=[{"role": "user", "content": build_cleanup_prompt(
                    list(group["content_raw"]))}],
            ),
        ))

    print(f"  cleaning: submitting {len(requests)} conversation phases to the "
          f"Batch API ({CLEANUP_MODEL}, effort={CLEANUP_EFFORT})", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"  cleaning: batch {batch.id}", flush=True)
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"    {batch.processing_status}: {counts.succeeded} done, "
              f"{counts.processing} processing, {counts.errored} errored",
              flush=True)
        time.sleep(poll_seconds)

    out = dialogue.copy()
    unusable = 0
    for result in client.messages.batches.results(batch.id):
        n = int(result.custom_id.removeprefix("phase-"))
        positions = index_of[n]
        texts = _cleaned_texts(result, len(positions))
        if texts is None:
            unusable += 1
            continue
        out.loc[positions, "content_clean"] = texts
        out.loc[positions, "cleaned"] = True

    out["n_words"] = [n_words(t) for t in out["content_clean"]]
    stats = {
        "cleanup_model": CLEANUP_MODEL,
        "cleanup_block_sha256": CLEANUP_BLOCK_SHA256,
        "phases_sent": len(requests),
        "phases_unusable_left_raw": unusable,
        "messages_cleaned": int(out["cleaned"].sum()),
        "messages_changed": int((out["content_clean"] != out["content_raw"]).sum()),
        "messages_emptied": int((out["content_clean"].str.strip() == "").sum()),
    }
    if unusable:
        print(f"  cleaning: {unusable} of {len(requests)} phases returned no "
              "usable answer; those keep their raw text", flush=True)
    return out, stats


def _cleaned_texts(result, expected: int) -> list[str] | None:
    """One phase's cleaned texts, or None when the answer cannot be trusted.

    The contract is one text per message, in order. A short, long or
    out-of-order list is rejected whole rather than aligned by guesswork --
    misaligning a cleaned line onto the wrong speaker would put words in a
    participant's mouth, which is the one failure this file must not have.
    """
    from src.baselines.prompting import extract_json_object

    if result.result.type != "succeeded":
        return None
    message = result.result.message
    if message.stop_reason == "refusal":
        return None
    text = next((b.text for b in message.content if b.type == "text"), "")
    obj = extract_json_object(text)
    items = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(items, list) or len(items) != expected:
        return None
    texts = [""] * expected
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        try:
            i = int(item.get("index"))
        except (TypeError, ValueError):
            return None
        if not 0 <= i < expected or i in seen:
            return None
        seen.add(i)
        texts[i] = str(item.get("text", "")).strip()
    return texts


CLEANUP_MODEL = "claude-opus-5"
CLEANUP_EFFORT = "low"          # a bounded rewrite against a fixed rule list
CLEANUP_MAX_TOKENS = 16000      # a long phase is ~200 messages of speech


def build_dialogue(raw: pd.DataFrame, corpus: str, *, clean: bool,
                   limit_groups: int | None = None) -> tuple[pd.DataFrame, dict]:
    """merge_runs, then the API cleaning pass when it is approved."""
    dialogue = merge_runs(raw, corpus)
    phases = dialogue.groupby(["conversation_id", "type"]).ngroups
    print(f"  {corpus}: {len(raw)} rows -> {len(dialogue)} merged messages "
          f"in {phases} phases across "
          f"{dialogue['conversation_id'].nunique()} conversations")
    stats = {
        "merged_messages": int(len(dialogue)),
        "phases": int(phases),
        "max_rows_merged_into_one_message": int(dialogue["n_merged"].max()),
        "messages_from_multiple_rows": int((dialogue["n_merged"] > 1).sum()),
    }
    if not clean:
        print("  cleaning SKIPPED (no --allow-network): dialogue.csv carries "
              "the merged raw text, uncleaned")
        return dialogue, {**stats, "fully_cleaned": False,
                          "messages_cleaned": 0}
    dialogue, cleanup_stats = clean_with_claude(dialogue,
                                                limit_groups=limit_groups)
    left_raw = len(uncleaned_phases(dialogue))
    if left_raw:
        print(f"  WARNING: {left_raw} phases are still uncleaned; re-run this "
              "command to finish them")
    return dialogue, {**stats, **cleanup_stats,
                      "fully_cleaned": left_raw == 0,
                      "phases_left_uncleaned": left_raw}


def prepare_corpus(corpus: str, *, clean: bool = False,
                   limit_groups: int | None = None,
                   overwrite: bool = False) -> dict:
    raw = load_raw(corpus)
    turns = build_turns(raw, corpus)
    sessions = build_sessions(raw)

    expected = EXPECTED[corpus]
    got = {
        "rows": len(raw),
        "user_turns": len(turns),
        "sessions": len(sessions),
        "participants": int(sessions["nickname"].nunique()),
    }
    assert got == expected, f"{corpus}: counts {got} != expected {expected}"

    out_dir = corpus_dir(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    turns.to_csv(out_dir / "turns.csv", index=False)
    sessions.to_csv(out_dir / "sessions.csv", index=False)

    dialogue_path = out_dir / "dialogue.csv"
    if dialogue_path.exists() and not overwrite:
        # The merge is deterministic, so an existing file is reused -- but a
        # reused file is NOT assumed to be clean. Whatever it is still missing
        # is cleaned now, because the alternative (keeping it as it is) means a
        # second run of this command with --allow-network does nothing at all,
        # and the file then reads as cleaned downstream when it is not.
        dialogue = pd.read_csv(dialogue_path).fillna({"content_clean": "",
                                                      "content_raw": ""})
        left = len(uncleaned_phases(dialogue))
        print(f"  {dialogue_path.name} exists, kept ({len(dialogue)} merged "
              f"messages, {left} phases not yet cleaned; --overwrite to "
              "rebuild from the raw CSV)")
        dialogue_stats = {"reused_existing": True,
                          "merged_messages": int(len(dialogue))}
        if clean and left:
            dialogue, cleanup_stats = clean_with_claude(
                dialogue, limit_groups=limit_groups)
            still = len(uncleaned_phases(dialogue))
            if still:
                print(f"  WARNING: {still} phases are still uncleaned; re-run "
                      "this command to finish them")
            dialogue.to_csv(dialogue_path, index=False)
            dialogue_stats |= {**cleanup_stats, "fully_cleaned": still == 0,
                               "phases_left_uncleaned": still}
        else:
            dialogue_stats |= {
                "fully_cleaned": left == 0,
                "messages_cleaned": int(dialogue["cleaned"].sum())}
    else:
        dialogue, dialogue_stats = build_dialogue(
            raw, corpus, clean=clean, limit_groups=limit_groups)
        dialogue.to_csv(dialogue_path, index=False)
        print(f"wrote data/conversation/{corpus}/dialogue.csv "
              f"({len(dialogue)} merged messages)")

    # Post-condition: load_raw already dropped these, so the residual must be 0.
    user_rows = raw[raw["source"] == "user"]
    exact_dupes = int(user_rows.duplicated(subset=RAW_COLUMNS).sum())
    assert exact_dupes == 0, (
        f"{corpus}: {exact_dupes} exact-duplicate user rows survived "
        "drop_exact_duplicates")
    audit = pd.read_csv(corpus_dir(corpus) / "dropped_duplicate_rows.csv")
    repeated = int((turns["content"]
                    == turns.groupby("conversation_id")["content"].shift()).sum())
    summary = {
        "raw_rows": got["rows"],
        "user_turns": got["user_turns"],
        "sessions": got["sessions"],
        "participants": got["participants"],
        "sessions_with_user_turns": int((sessions["n_user_turns"] > 0).sum()),
        "turns_ge_4_words": int((turns["n_words"] >= 4).sum()),
        "dropped_duplicate_rows": int(len(audit)),
        "dropped_duplicate_user_rows": int((audit["source"] == "user").sum()),
        "residual_exact_duplicate_user_rows": exact_dupes,
        "consecutive_identical_content_turns": repeated,
        "dialogue": dialogue_stats,
    }
    record_decision(f"prepare_data.{corpus}", summary)

    print(f"wrote data/conversation/{corpus}/turns.csv     ({len(turns)} user turns)")
    print(f"wrote data/conversation/{corpus}/sessions.csv  ({len(sessions)} sessions)")
    print(f"  counts match: {got}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", choices=CORPORA, default=None,
                    help="prepare one corpus (default: both)")
    ap.add_argument("--allow-network", action="store_true",
                    help="approve sending participant speech to the Claude "
                         "API for the cleaning pass; without it dialogue.csv "
                         "is written merged but uncleaned")
    ap.add_argument("--limit-groups", type=int, default=None, metavar="N",
                    help="clean only the first N conversation phases per "
                         "corpus (a smoke run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild dialogue.csv even if it exists (re-runs the "
                         "cleaning pass, and so re-spends the API call)")
    args = ap.parse_args()

    for corpus in ([args.corpus] if args.corpus else CORPORA):
        prepare_corpus(corpus, clean=args.allow_network,
                       limit_groups=args.limit_groups,
                       overwrite=args.overwrite)
        print()
    print("sanity checks passed")


if __name__ == "__main__":
    sys.exit(main())
