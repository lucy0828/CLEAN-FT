"""LonXplain raw-data processing: everything Stages 1 and 2 read.

Independent and runnable — imports nothing from the rest of src/:

    uv run python -m src.utils.data_prep                 # posts.csv + sentences.json
    uv run python -m src.utils.data_prep --encode --model gemma-2-2b --allow-model-process
    uv run python -m src.utils.data_prep --encode --model llama31-8b --allow-model-process

Processed data produced (and the loaders the stages read it back with):

1. data/lonxplain/posts.csv       — ftfy-repaired posts, present/absent labels,
                                    psychiatrist spans aligned to character
                                    offsets, stratified 10-fold split (seed 42).
2. data/lonxplain/sentences.json  — spaCy sentence spans tiling each post.
3. cache/activations/{model}/lonxplain_layer{L:02d}.parquet — per-token hidden
                                    states at depth fractions {0.25..0.75};
                                    chat-template tokens dropped before writing.

Loaders: `words`/`normalize_word` (the study's single word tokenizer),
`load_corpus`, `load_layer` + word/sentence/post pooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "activations"

MODEL_REGISTRY = {
    "llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-2b": "google/gemma-2-2b-it",
}

# Cached depth fractions; the 0.65 working layer sits on this grid.
# --fractions selects a subset.
LAYER_FRACTIONS = [0.25, 0.4, 0.5, 0.65, 0.75]


def _config() -> dict:
    """configs/default.yaml, read locally so this module stays independent."""
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


# ============================================================ word tokenizer

_WORD_RE = re.compile(r"\S+")
_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def words(text: str) -> list[dict]:
    """Split text into words with character spans.

    A word is a maximal run of non-whitespace. Returns a list of
    {"text", "start", "end", "norm"} dicts. `norm` is the scoring form:
    lowercased with leading/trailing punctuation stripped (lowercase,
    punctuation stripped, no stemming). Words that normalize to the empty
    string (pure punctuation) keep norm="" and are excluded from ROUGE sets.
    """
    out = []
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        out.append(
            {
                "text": raw,
                "start": m.start(),
                "end": m.end(),
                "norm": _PUNCT_RE.sub("", raw).lower(),
            }
        )
    return out


def normalize_word(w: str) -> str:
    return _PUNCT_RE.sub("", w).lower()


# ========================================================== chat template

def build_prompt(tokenizer, post: str) -> tuple[str, int, int]:
    """Text to encode + the post's character span inside it.

    Every studied backbone is an open decoder LLM, so the post is wrapped in
    the model's chat template; template tokens are dropped before pooling.
    """
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": post}],
        tokenize=False,
        add_generation_prompt=False,
    )
    i = full.find(post)
    if i < 0:
        # Some chat templates (gemma-2) trim the message content, so deletion
        # variants that carry leading/trailing whitespace are not found
        # verbatim. Internal whitespace is preserved,
        # so matching the stripped text recovers the exact span.
        post = post.strip()
        i = full.find(post)
    assert i >= 0, "post text not found verbatim in chat template output"
    return full, i, i + len(post)


# ========================================== corpus + activation loaders

def load_corpus(corpus: str):
    """(posts, sentences) for a corpus under data/.

    LonXplain posts carry: id, text, label ("present"/"absent"), fold,
    span_start, span_end.
    """
    posts = pd.read_csv(DATA / corpus / "posts.csv")
    with open(DATA / corpus / "sentences.json") as f:
        sentences = json.load(f)
    return posts, sentences


def load_layer(model: str, corpus: str, layer: int) -> dict[str, dict]:
    """Read one layer's parquet into {post_id: {"vecs": [T,H] f32, "offsets": [T,2]}}.

    The vectors are a fixed-size list column, so they come out as one flat
    buffer that reshapes without copying — stacking them row by row instead
    costs minutes per fold on a corpus this size.
    """
    path = CACHE / model / f"{corpus}_layer{layer:02d}.parquet"
    table = pq.read_table(path)
    pid = table["post_id"].to_numpy(zero_copy_only=False)
    start = table["start"].to_numpy()
    end = table["end"].to_numpy()
    vec_col = table["vec"].combine_chunks()
    H = vec_col.type.list_size
    vecs = (
        vec_col.values.to_numpy(zero_copy_only=False)
        .reshape(len(vec_col), H)
        .astype(np.float32)
    )
    del table, vec_col

    out = {}
    order = np.argsort(pid, kind="stable")
    pid, start, end, vecs = pid[order], start[order], end[order], vecs[order]
    boundaries = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        out[pid[a]] = {
            "vecs": vecs[a:b],
            "offsets": np.stack([start[a:b], end[a:b]], axis=1),
        }
    return out


def post_vector(acts: dict) -> np.ndarray:
    return acts["vecs"].mean(axis=0)


def token_sentence_ids(offsets: np.ndarray, sent_spans: list[list[int]]) -> np.ndarray:
    """Assign each token to exactly one sentence by its start character."""
    starts = np.array([s for s, _ in sent_spans])
    idx = np.searchsorted(starts, offsets[:, 0], side="right") - 1
    return np.clip(idx, 0, len(sent_spans) - 1)


def sentence_vectors(acts: dict, sent_spans: list[list[int]]):
    """Return ([S,H] mean vectors, [S] token counts); empty sentences get NaN."""
    sids = token_sentence_ids(acts["offsets"], sent_spans)
    H = acts["vecs"].shape[1]
    vecs = np.full((len(sent_spans), H), np.nan, dtype=np.float32)
    counts = np.zeros(len(sent_spans), dtype=np.int64)
    for s in range(len(sent_spans)):
        mask = sids == s
        counts[s] = mask.sum()
        if counts[s]:
            vecs[s] = acts["vecs"][mask].mean(axis=0)
    return vecs, counts


def word_vectors(acts: dict, post_words: list[dict]) -> np.ndarray:
    """[W,H] mean of tokens overlapping each word's char span; NaN if none."""
    H = acts["vecs"].shape[1]
    out = np.full((len(post_words), H), np.nan, dtype=np.float32)
    tok_s, tok_e = acts["offsets"][:, 0], acts["offsets"][:, 1]
    for i, w in enumerate(post_words):
        mask = (tok_s < w["end"]) & (tok_e > w["start"])
        if mask.any():
            out[i] = acts["vecs"][mask].mean(axis=0)
    return out


def verify_identity(model: str, corpus: str, layer: int, n_check: int = 50) -> float:
    """Sanity check: post vector == token-count-weighted mean of sentence vectors."""
    acts = load_layer(model, corpus, layer)
    with open(DATA / corpus / "sentences.json") as f:
        sentences = json.load(f)
    worst = 0.0
    for pid in list(acts)[:n_check]:
        pv = post_vector(acts[pid])
        sv, counts = sentence_vectors(acts[pid], sentences[pid])
        keep = counts > 0
        recon = (sv[keep] * counts[keep, None]).sum(axis=0) / counts.sum()
        worst = max(worst, float(np.abs(pv - recon).max()))
    return worst


# =============================================== raw-data preparation (posts / sentences / activation cache)

# =========================================================== 1. posts.csv

def fix_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ftfy.fix_text(s)


_DROP_CHARS = set("'’‘`´“”\"")


def _norm_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase, drop apostrophes/quotes, collapse whitespace runs to one
    space. Returns (normalized string, map from normalized index -> original
    index) so matches can be projected back to original character offsets."""
    out, idx_map = [], []
    prev_space = True
    for i, ch in enumerate(s):
        if ch in _DROP_CHARS:
            continue
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx_map.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        idx_map.pop()
    return "".join(out), idx_map


def align_span(text: str, span: str) -> tuple[int, int, int]:
    """Return (start, end, n_occurrences) of span in text, first match.

    Tries exact match, then a normalized match that is insensitive to case,
    apostrophes/quotes, and whitespace runs (the released spans are lightly
    edited: "didnt wish me" vs post "didn't wish me"). Matches found in
    normalized space are projected back to original offsets. Returns
    (-1, -1, 0) when the span cannot be located (true paraphrases).
    """
    if not isinstance(span, str) or not span.strip():
        return -1, -1, 0
    span = span.strip()

    n = text.count(span)
    if n > 0:
        i = text.find(span)
        return i, i + len(span), n

    norm_text, idx_map = _norm_with_map(text)
    norm_span, _ = _norm_with_map(span)
    if not norm_span:
        return -1, -1, 0
    n = norm_text.count(norm_span)
    if n > 0:
        j = norm_text.find(norm_span)
        start = idx_map[j]
        end = idx_map[j + len(norm_span) - 1] + 1
        return start, end, n

    # Last resort for lightly paraphrased spans ("distanced emotionally" vs
    # post "distance myself emotionally"): best contiguous word window by
    # SequenceMatcher ratio in normalized space; accepted only at >= 0.85.
    from difflib import SequenceMatcher

    _words = words

    NEG = {"no", "not", "never", "none", "nobody", "nothing", "dont", "cant",
           "wont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
           "havent", "hasnt", "couldnt", "shouldnt", "wouldnt", "aint"}

    def has_neg(norm_s: str) -> bool:
        return any(w in NEG for w in norm_s.split())

    span_words = norm_span.split()
    post_words = _words(text)
    span_neg = has_neg(norm_span)
    best_ratio, best = 0.0, None
    for wlen in range(max(1, len(span_words) - 2), len(span_words) + 3):
        for i in range(0, len(post_words) - wlen + 1):
            a, b = post_words[i]["start"], post_words[i + wlen - 1]["end"]
            cand, _ = _norm_with_map(text[a:b])
            # A span with negation must match a window with negation — a
            # high-similarity window that drops the "don't" inverts meaning.
            if span_neg != has_neg(cand):
                continue
            ratio = SequenceMatcher(None, cand, norm_span).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (a, b)
    if best is not None and best_ratio >= 0.85:
        return best[0], best[1], 1

    return -1, -1, 0


def prepare_lonxplain(seed: int) -> pd.DataFrame:
    raw = pd.read_csv(DATA / "lonxplain" / "raw" / "Lonesomenessdataset.csv")
    df = pd.DataFrame(
        {
            "id": ["lx%04d" % i for i in raw["Unnamed: 0"]],
            "text": raw["text"].map(fix_text),
            "label": raw["tb"].map({1: "present", 0: "absent"}),
            "span_text": raw["tb_exp"].map(fix_text),
        }
    )

    assert len(df) == 3522, f"expected 3522 rows, got {len(df)}"
    counts = df["label"].value_counts().to_dict()
    assert counts == {"present": 1927, "absent": 1595}, f"label counts {counts}"

    # Spans belong to present posts only; log any anomaly.
    stray = int(((df["label"] == "absent") & df["span_text"].notna()).sum())
    df.loc[df["label"] == "absent", "span_text"] = np.nan

    starts, ends, multi, unaligned = [], [], [], []
    for _, row in df.iterrows():
        if row["label"] != "present" or not isinstance(row["span_text"], str):
            starts.append(-1)
            ends.append(-1)
            continue
        s, e, n = align_span(row["text"], row["span_text"])
        starts.append(s)
        ends.append(e)
        if n > 1:
            multi.append(row["id"])
        if n == 0:
            unaligned.append(row["id"])
    df["span_start"] = starts
    df["span_end"] = ends

    # Sanity: aligned spans fall inside the post.
    ok = (df["span_start"] < 0) | (
        (df["span_start"] >= 0) & (df["span_end"] <= df["text"].str.len())
    )
    assert ok.all(), "span offsets escape their post"

    n_present_missing_span = int(
        (df["label"].eq("present") & df["span_text"].isna()).sum()
    )
    print(
        "prepare_data.lonxplain:",
        {
            "rows": len(df),
            "present": counts["present"],
            "absent": counts["absent"],
            "spans_on_absent_posts_dropped": stray,
            "present_posts_without_span": n_present_missing_span,
            "spans_with_multiple_matches_first_taken": len(multi),
            "spans_unalignable": len(unaligned),
            "unalignable_ids": unaligned[:50],
        },
    )

    # Stratified 10-fold assignment, seed 42.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, test_idx) in enumerate(skf.split(df, df["label"])):
        fold[test_idx] = k
    df["fold"] = fold
    assert (df["fold"] >= 0).all()

    return df


# ====================================================== 2. sentences.json


def segment_posts(texts: dict[str, str]) -> dict[str, list[list[int]]]:
    import spacy

    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "tagger"])
    out = {}
    ids = list(texts)
    for pid, doc in zip(ids, nlp.pipe([texts[i] for i in ids], batch_size=64)):
        text = texts[pid]
        # spaCy sentence char spans; extend each end to the next start so the
        # spans tile the post (whitespace attaches to the preceding sentence).
        spans = [[s.start_char, s.end_char] for s in doc.sents]
        if not spans:
            spans = [[0, len(text)]]
        spans[0][0] = 0
        for i in range(len(spans) - 1):
            spans[i][1] = spans[i + 1][0]
        spans[-1][1] = len(text)

        # Byte-for-byte reconstruction check.
        rebuilt = "".join(text[a:b] for a, b in spans)
        assert rebuilt == text, f"reconstruction failed for {pid}"
        out[pid] = spans
    return out


# =================================================== 3. activation cache


def encode_corpus(model_name: str, corpus: str, batch_size: int, max_len: int,
                  fractions: list[float] | None = None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_REGISTRY[model_name]
    posts = pd.read_csv(DATA / corpus / "posts.csv")
    out_dir = CACHE / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    fractions = fractions or LAYER_FRACTIONS
    layers = sorted({int(round(f * num_layers)) for f in fractions})

    meta = {
        "model_name": model_name,
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "layer_fractions": fractions,
        "layers": layers,
        "dtype": "float16",
        "kind": "decoder",
        "max_len": max_len,
    }
    with open(out_dir / f"{corpus}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"model {model_id}: {num_layers} layers, keeping {layers}")

    schema = pa.schema(
        [
            ("post_id", pa.string()),
            ("token_idx", pa.int32()),
            ("start", pa.int32()),
            ("end", pa.int32()),
            ("vec", pa.list_(pa.float16(), hidden_dim)),
        ]
    )
    writers = {
        L: pq.ParquetWriter(out_dir / f"{corpus}_layer{L:02d}.parquet", schema)
        for L in layers
    }

    # Pre-tokenize everything (cheap) so batches can be sorted by length.
    records = []
    for pid, text in zip(posts["id"], posts["text"].fillna("")):
        full, p0, p1 = build_prompt(tokenizer, text)
        enc = tokenizer(
            full,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        keep = [
            (ti, a - p0, b - p0)
            for ti, (a, b) in enumerate(enc["offset_mapping"])
            if a < p1 and b > p0 and b > a  # token overlaps the post text
        ]
        assert keep, f"no post tokens for {pid}"
        records.append((pid, enc["input_ids"], keep))
    records.sort(key=lambda r: len(r[1]))

    done = 0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            maxlen = max(len(r[1]) for r in batch)
            # Explicit None check: BERT's pad id is 0, which `or` would treat
            # as missing and replace with a nonexistent eos.
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, (_, ids, _) in enumerate(batch):
                input_ids[j, : len(ids)] = torch.tensor(ids)
                attn[j, : len(ids)] = 1
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)

            out = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True
            )
            for L in layers:
                h = out.hidden_states[L].to(torch.float16).cpu()  # [B, T, H]
                cols = {"post_id": [], "token_idx": [], "start": [], "end": [], "vec": []}
                for j, (pid, _, keep) in enumerate(batch):
                    for k, (ti, a, b) in enumerate(keep):
                        cols["post_id"].append(pid)
                        cols["token_idx"].append(k)
                        cols["start"].append(max(a, 0))
                        cols["end"].append(b)
                        cols["vec"].append(h[j, ti].numpy())
                writers[L].write_table(pa.table(cols, schema=schema))
            del out
            done += len(batch)
            if done % (batch_size * 8) == 0 or done == len(records):
                print(f"  encoded {done}/{len(records)} posts", flush=True)

    for w in writers.values():
        w.close()
    print(f"wrote {len(layers)} layer files to {out_dir}")


# ================================================================== driver


def _require_model_process(cfg: dict, cli_override: bool, command: str):
    """Refuse model work unless config or an explicit CLI override allows it."""
    configured = bool(cfg.get("execution", {}).get("allow_model_process", False))
    if not (configured or cli_override):
        raise RuntimeError(
            f"{command} would load a study model; re-run with "
            "--allow-model-process (or set execution.allow_model_process).")


def main():
    cfg = _config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encode", action="store_true",
                    help="cache per-token activations (GPU) instead of "
                         "building posts/sentences (CPU)")
    ap.add_argument("--model", default="gemma2",
                    help="--encode: backbone (gemma2 or llama)")
    ap.add_argument("--corpus", default="lonxplain")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--fractions", type=float, nargs="+", default=None,
                    help="--encode: subset of the cached depth fractions")
    ap.add_argument("--allow-model-process", action="store_true")
    args = ap.parse_args()

    if args.encode:
        _require_model_process(cfg, args.allow_model_process,
                               "src.utils.data_prep --encode")
        model = {"gemma2": "gemma-2-2b", "llama": "llama31-8b"}.get(
            args.model.lower(), args.model)
        if model not in MODEL_REGISTRY:
            raise SystemExit(f"unknown --model {args.model!r}; use gemma2 or llama")
        encode_corpus(model, args.corpus,
                      args.batch_size, args.max_len, args.fractions)
        return

    lx = prepare_lonxplain(seed=cfg["seed"])
    lx.to_csv(DATA / "lonxplain" / "posts.csv", index=False)
    print(f"wrote data/lonxplain/posts.csv  ({len(lx)} rows)")
    posts = {row["id"]: row["text"] for _, row in lx.iterrows()}
    spans = segment_posts(posts)
    with open(DATA / "lonxplain" / "sentences.json", "w") as f:
        json.dump(spans, f)
    print(f"wrote data/lonxplain/sentences.json ({len(spans)} posts)")


if __name__ == "__main__":
    main()
