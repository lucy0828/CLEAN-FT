"""Prompting the backbone instead of reading it.

Not run directly — src/stage1/train.py drives it.

The detection prompt (this module builds the text from configs/prompts.yaml)
is zero-shot: the pinned clinical rubric, the annotator role and reasoning
pipeline, and the output contract, with no labelled exemplars. The model
answers with one JSON object carrying its reasoning, the extracted cause and
consequence spans, the active dimensions, which rules fired, and the label.

It is scored two ways:

  detection — P(Present) / (P(Present) + P(Absent)) from the first-token
              logits. The contract puts "thought_process" first, so there is
              no fixed prefix to teacher-force: the answer is generated
              greedily first, then the model's OWN completion is replayed up
              to the opening quote of "lonesomeness_label" and the logits are
              read there. The score is therefore conditioned on the same
              reasoning that produced the reported label, rather than on a
              label the model was never allowed to think about. A completion
              that never emits the key scores NaN and is dropped from AUROC
              rather than guessed at.
  evidence  — every extracted span is matched back onto post word positions;
              words that are not in the post are discarded and the discard
              rate is logged.
"""

import hashlib
import json
import re

import numpy as np
import pandas as pd
import yaml

from src.utils.common import ROOT, record_decision

from src.utils.data_prep import normalize_word, words


def extract_json_object(text: str) -> dict:
    """The first complete JSON object in a free-form completion, or {}.

    Brace-matched rather than regex-matched: the answer contract nests
    `rules_triggered` inside the top-level object, and a non-greedy `\{.*?\}`
    stops at the inner object's closing brace and yields invalid JSON. Quoted
    braces and escapes are skipped so a brace inside an extracted span cannot
    end the scan early.
    """
    start = text.find("{")
    if start < 0:
        return {}
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return {}
                    return obj if isinstance(obj, dict) else {}
    return {}


def _string_list(value) -> list[str]:
    """Coerce one contract field to a list of non-empty strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


LABEL_KEY = "lonesomeness_label"
RULE_KEYS = ("temporal_rule_applied", "appraisal_rule_applied",
             "conflict_exclusion_applied", "depression_exclusion_applied")


def parse_answer(raw: str) -> dict:
    """The answer contract out of one generated completion; tolerant.

    Returns `label` (None when nothing parses -- never guessed), the two span
    lists, the active dimensions, the four rule flags, and the reasoning.
    """
    obj = extract_json_object(raw)
    label = str(obj.get(LABEL_KEY, "")).strip().lower()
    if label not in ("present", "absent"):
        # The object did not parse, or carried no usable label. Fall back to
        # the key on its own so a truncated completion is not thrown away.
        m = re.search(rf'"?{LABEL_KEY}"?\s*[:=]?\s*"?(present|absent)',
                      raw, flags=re.IGNORECASE)
        label = m.group(1).lower() if m else ""
    rules = obj.get("rules_triggered")
    rules = rules if isinstance(rules, dict) else {}
    return {
        "label": {"present": "Present", "absent": "Absent"}.get(label),
        "causes": _string_list(obj.get("extracted_causes")),
        "consequences": _string_list(obj.get("extracted_consequences")),
        "dimensions": _string_list(obj.get("active_dimensions")),
        "rules": {k: bool(rules.get(k, False)) for k in RULE_KEYS},
        "thought_process": str(obj.get("thought_process", "")).strip(),
    }


def evidence_to_word_indices(post_text: str, evidence: str) -> tuple[list[int], int]:
    """Map quoted evidence words onto post word indices. The quote is matched
    as a contiguous normalized word sequence first (first occurrence, like
    gold-span alignment in src.utils.data_prep.align_span); only if no
    contiguous match exists
    does it fall back to per-word first-unused-occurrence matching, so a
    phrase quoted from late in the post lands at its true position rather
    than on scattered earlier duplicates. Returns (indices, n_discarded)."""
    post_words = words(post_text)
    ev_norms = [n for n in (normalize_word(t) for t in evidence.split()) if n]
    if not ev_norms:
        return [], 0

    post_norms = [w["norm"] for w in post_words]
    for start in range(len(post_norms) - len(ev_norms) + 1):
        if post_norms[start : start + len(ev_norms)] == ev_norms:
            return list(range(start, start + len(ev_norms))), 0

    used = set()
    idx, discarded = [], 0
    for norm in ev_norms:
        hit = next(
            (
                i
                for i, w in enumerate(post_words)
                if i not in used and w["norm"] == norm
            ),
            None,
        )
        if hit is None:
            discarded += 1
        else:
            used.add(hit)
            idx.append(hit)
    return idx, discarded


def spans_to_word_indices(post_text: str, spans: list[str]) -> tuple[list[int], int]:
    """Union of the word positions covered by every extracted span.

    Each span is aligned independently (a span is a contiguous quote; the set
    of spans is not), then merged, so two spans quoting the same words do not
    inflate the evidence budget. Discards are summed across spans.
    """
    covered: set[int] = set()
    discarded = 0
    for span in spans:
        idx, disc = evidence_to_word_indices(post_text, span)
        covered.update(idx)
        discarded += disc
    return sorted(covered), discarded


def answer_prompts(model, tok, prompts: list[str], texts: list[str]) -> dict:
    """Answer and score a batch of prompts.

    Generation runs FIRST: the contract puts the reasoning before the label,
    so the score has to be read against the model's own completion (see the
    module docstring). Returns parallel lists ready for a CSV.
    """
    print("generating answers", flush=True)
    raw = batched_generate(model, tok, prompts)

    parsed = [parse_answer(r) for r in raw]

    print(f"scoring P(Present) for {len(prompts)} prompts", flush=True)
    scores = batched_label_scores(model, tok, prompts, raw)

    ev_indices, discards, evidences = [], [], []
    for text, a in zip(texts, parsed):
        spans = a["causes"] + a["consequences"]
        idx, disc = spans_to_word_indices(text, spans)
        ev_indices.append(json.dumps(idx))
        discards.append(disc)
        evidences.append(" | ".join(spans))
    return {
        "present_score": scores,
        "pred_label": [a["label"] for a in parsed],
        "evidence": evidences,
        "evidence_word_indices": ev_indices,
        "discarded_words": discards,
        "extracted_causes": [json.dumps(a["causes"]) for a in parsed],
        "extracted_consequences": [json.dumps(a["consequences"]) for a in parsed],
        "active_dimensions": [json.dumps(a["dimensions"]) for a in parsed],
        "rules_triggered": [json.dumps(a["rules"]) for a in parsed],
        "thought_process": [a["thought_process"] for a in parsed],
    }


# ------------------------------------------------------- API-backbone arm
#
# The same prompt, the same construct block and the same answer contract as the
# local arms -- only the weights differ, and they are weights this study cannot
# open. Two consequences follow, and both are deliberate rather than
# oversights:
#
#   1. No `present_score`. A local backbone is scored by reading the token
#      distribution at the label position (`batched_label_scores`); the API
#      returns a decision and no distribution, so the column is written NaN.
#      The arm therefore publishes F1, FPR and FNR and no AUROC -- the exact
#      mirror of a probe, which publishes a ranking and no decision. Stage 3's
#      `score_claude` records the same constraint for the same reason.
#   2. The answer contract is enforced by the API's structured outputs rather
#      than by this file's tolerant parser, so a malformed answer is a
#      validated retry inside the API instead of a dropped row here. The
#      instruction text is byte-identical to what the local arms are given;
#      what differs is that the schema is also machine-enforced.
#
# Batch rather than live requests: there is no latency requirement over 3,522
# posts and batch is half price.

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {"type": "string"},
        "extracted_causes": {"type": "array", "items": {"type": "string"}},
        "extracted_consequences": {"type": "array",
                                   "items": {"type": "string"}},
        "active_dimensions": {"type": "array", "items": {"type": "string"}},
        "rules_triggered": {
            "type": "object",
            "properties": {k: {"type": "boolean"} for k in RULE_KEYS},
            "required": list(RULE_KEYS),
            "additionalProperties": False,
        },
        LABEL_KEY: {"type": "string", "enum": ["Present", "Absent"]},
    },
    "required": ["thought_process", "extracted_causes",
                 "extracted_consequences", "active_dimensions",
                 "rules_triggered", LABEL_KEY],
    "additionalProperties": False,
}


def answer_prompts_api(prompts: list[str], texts: list[str], model: str,
                       *, poll_seconds: int = 30) -> dict:
    """`answer_prompts` for an API backbone: same columns, no score column."""
    import time

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from src.utils.common import load_config, load_dotenv

    cfg = load_config().get("prompting_api", {})
    effort = cfg.get("effort", "low")
    max_tokens = int(cfg.get("max_tokens", 4096))

    load_dotenv()
    client = anthropic.Anthropic()
    requests = [
        Request(
            custom_id=f"post-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema",
                               "schema": DETECTION_SCHEMA},
                },
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, prompt in enumerate(prompts)
    ]

    print(f"  submitting {len(requests)} posts to the Batch API "
          f"({model}, effort={effort})", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id}", flush=True)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"    {batch.processing_status}: {counts.succeeded} done, "
              f"{counts.processing} processing, {counts.errored} errored",
              flush=True)
        time.sleep(poll_seconds)

    # Batch results arrive in any order, so they are keyed by custom_id and
    # never by position. A request that errored, was refused, or returned no
    # text leaves its slot empty, and parse_answer maps that to label None --
    # the same "did not parse" marker the local arms use, counted downstream
    # as n_unparsed and excluded from the decision metrics rather than being
    # silently scored as absent.
    raw = [""] * len(prompts)
    for result in client.messages.batches.results(batch.id):
        i = int(result.custom_id.removeprefix("post-"))
        if result.result.type != "succeeded":
            continue
        message = result.result.message
        if message.stop_reason == "refusal":
            continue
        raw[i] = next((b.text for b in message.content if b.type == "text"), "")

    parsed = [parse_answer(r) for r in raw]
    n_empty = sum(1 for r in raw if not r)
    if n_empty:
        print(f"  {n_empty} of {len(prompts)} posts returned no usable answer; "
              "recorded unparsed, never guessed", flush=True)

    ev_indices, discards, evidences = [], [], []
    for text, a in zip(texts, parsed):
        spans = a["causes"] + a["consequences"]
        idx, disc = spans_to_word_indices(text, spans)
        ev_indices.append(json.dumps(idx))
        discards.append(disc)
        evidences.append(" | ".join(spans))
    return {
        "present_score": [float("nan")] * len(prompts),
        "pred_label": [a["label"] for a in parsed],
        "evidence": evidences,
        "evidence_word_indices": ev_indices,
        "discarded_words": discards,
        "extracted_causes": [json.dumps(a["causes"]) for a in parsed],
        "extracted_consequences": [json.dumps(a["consequences"]) for a in parsed],
        "active_dimensions": [json.dumps(a["dimensions"]) for a in parsed],
        "rules_triggered": [json.dumps(a["rules"]) for a in parsed],
        "thought_process": [a["thought_process"] for a in parsed],
    }


# ------------------------------------------------ causal-LM plumbing
# Formerly src/utils/llm.py: load a backbone, apply its chat template,
# score or generate in batches.


def load_llm(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.data_prep import MODEL_REGISTRY

    model_id = MODEL_REGISTRY[model_name]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tok


def chat_text(tok, user_msg: str, assistant_prefix: str = "") -> str:
    msgs = [{"role": "user", "content": user_msg}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        text = tok.apply_chat_template(msgs, **kwargs)
    return text + assistant_prefix


LABEL_ANCHOR = re.compile(rf'"{LABEL_KEY}"\s*:\s*"')


def label_score_prefixes(raw_answers: list[str]) -> list[str | None]:
    """Each completion cut just after the label's opening quote.

    That character is the one position where the next token is exactly
    "Present" or "Absent", with all of the model's reasoning already in
    context. None marks a completion that never emitted the key.
    """
    out: list[str | None] = []
    for raw in raw_answers:
        m = LABEL_ANCHOR.search(raw)
        out.append(raw[: m.end()] if m else None)
    return out


def batched_label_scores(model, tok, prompts: list[str],
                         raw_answers: list[str]) -> np.ndarray:
    """P(Present)/(P(Present)+P(Absent)), replayed against each completion.

    NaN where the completion carried no label key -- src.stage1.detection drops
    non-finite scores from AUROC rather than reading a missing answer as a
    confident "absent".
    """
    import torch

    present_id = tok.encode("Present", add_special_tokens=False)[0]
    absent_id = tok.encode("Absent", add_special_tokens=False)[0]
    prefixes = label_score_prefixes(raw_answers)
    scores = np.full(len(prompts), np.nan)
    scorable = [i for i, pre in enumerate(prefixes) if pre is not None]
    B = 16
    with torch.no_grad():
        for i in range(0, len(scorable), B):
            chunk = scorable[i : i + B]
            batch = [
                chat_text(tok, prompts[j], assistant_prefix=prefixes[j])
                for j in chunk
            ]
            # add_special_tokens=False: the chat template already carries BOS
            # (Llama would otherwise get a double BOS; matches
            # src.utils.data_prep.encode_corpus).
            enc = tok(batch, return_tensors="pt", padding=True, padding_side="left",
                      add_special_tokens=False).to(model.device)
            # logits_to_keep=1: only the last position is read, and the full
            # (batch, seq, vocab) logits tensor OOMs an 80 GB card on 8B
            # backbones with long prompts.
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            two = logits[:, [present_id, absent_id]].float()
            probs = torch.softmax(two, dim=-1)[:, 0].cpu().numpy()
            for k, j in enumerate(chunk):
                scores[j] = probs[k]
    n_missing = len(prompts) - len(scorable)
    if n_missing:
        print(f"  {n_missing} completion(s) carried no {LABEL_KEY}; scored NaN",
              flush=True)
    return scores


def batched_generate(model, tok, prompts: list[str], max_new_tokens=640,
                     batch_size: int = 16) -> list[str]:
    """Greedy batched generation.

    Batching matters: thousands of generations one at a time is hours of
    wall-clock on an 8B model, and minutes in batches. `batch_size` defaults
    to the 16 the published prompting rows were generated with; greedy
    decoding makes the completion independent of the batch it sat in up to
    padding-side numerics.
    """
    import torch

    outs = []
    B = int(batch_size)
    with torch.no_grad():
        for i in range(0, len(prompts), B):
            batch = [chat_text(tok, p) for p in prompts[i : i + B]]
            enc = tok(batch, return_tensors="pt", padding=True, padding_side="left",
                      add_special_tokens=False).to(model.device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            for j in range(len(batch)):
                outs.append(
                    tok.decode(
                        gen[j, enc["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                )
            if (i // B) % 20 == 0:
                print(f"  generated {i + len(batch)}/{len(prompts)}", flush=True)
    return outs


# ==================================================== pinned prompt texts

PROMPTS_PATH = ROOT / "configs" / "prompts.yaml"

with open(PROMPTS_PATH) as _f:
    _P = yaml.safe_load(_f)

# ----------------------------------------------------------------- the texts

CLINICAL_RUBRIC = _P["clinical_rubric"]
TASK = _P["detection_task"]
OUTPUT = _P["detection_output"]
# Stage 3 screening: the construct's definition sentence, the screening role,
# and a one-digit answer. Deliberately short and deliberately rule-free --
# Stage 3 selects clinician attention rather than labelling a benchmark. See
# the header of configs/prompts.yaml.
SCREENING_PROMPT = _P["screening_prompt"]
SCREENING_OUTPUT = _P["screening_output"]
# Stage 3 transcript repair, used by src.stage3.prepare_conversations.
CLEANUP_TASK = _P["cleanup_task"]
CLEANUP_OUTPUT = _P["cleanup_output"]

# ------------------------------------------------------------- the guardrail

CLINICAL_RUBRIC_SHA256 = (
    "312623f397f1b6e484662ed57593f6cc58c934e377ef380e506692b2ba8b841c"
)
SCREENING_PROMPT_SHA256 = (
    "ba980acc6f28bc26fca214d2cdf34f1d7eb9091787c2cbf95d6040440260a18c"
)
SCREENING_BLOCK_SHA256 = (
    "9271673af0ce6bf21533e608ce0a28d35fe2c81dc5f114967d1c0a4c5838131a"
)
CLEANUP_BLOCK_SHA256 = (
    "a3755c881b84fb436e41561a6fa7d2523c9868dec4febffb24fda31c88ad1680"
)

for _text, _digest, _name in (
    (CLINICAL_RUBRIC, CLINICAL_RUBRIC_SHA256, "clinical rubric"),
    (SCREENING_PROMPT, SCREENING_PROMPT_SHA256, "screening prompt"),
    (SCREENING_PROMPT + SCREENING_OUTPUT, SCREENING_BLOCK_SHA256,
     "screening block"),
    (CLEANUP_TASK + CLEANUP_OUTPUT, CLEANUP_BLOCK_SHA256, "cleanup block"),
):
    assert hashlib.sha256(_text.encode()).hexdigest() == _digest, (
        f"{_name} changed in configs/prompts.yaml without updating its digest"
    )


# ------------------------------------------------------- detection prompts


def build_detection_prompt(post: str) -> str:
    """Assemble one Stage 1 detection prompt: pinned rubric, task, output
    contract. Zero-shot -- no labelled exemplars, so the arm consumes no
    LonXplain label and matches the Stage 3 turn prompt in construction.
    """
    return "\n\n".join(
        [CLINICAL_RUBRIC, TASK, OUTPUT.replace("{post}", post)]
    )


# ------------------------------------------- the Stage 3 conversational prompt


def build_screening_prompt(exchange: str) -> str:
    """The Stage 3 screening prompt for ONE conversation phase.

    Not the Stage 1 rubric, and not per turn. Stage 3 asks whether a clinician
    should read this exchange, not whether a reply meets the annotation
    standard, so it carries the definition sentence alone -- none of Stage 1's
    four annotation rules -- and reads the whole phase, both speakers, in
    order. The answer is one digit.
    """
    return "\n\n".join([SCREENING_PROMPT, exchange.strip(), SCREENING_OUTPUT])


def build_cleanup_prompt(messages: list[str]) -> str:
    """The Stage 3 transcript-repair prompt for one conversation phase."""
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(messages))
    return "\n\n".join([CLEANUP_TASK, "Messages:", numbered, CLEANUP_OUTPUT])


def check_construct_hashes():
    """Record the pinned digests at the start of every construct-carrying run."""
    record_decision(
        "construct.sha256",
        {"rubric": CLINICAL_RUBRIC_SHA256,
         "screening_prompt": SCREENING_PROMPT_SHA256,
         "screening_block": SCREENING_BLOCK_SHA256,
         "cleanup_block": CLEANUP_BLOCK_SHA256},
    )
