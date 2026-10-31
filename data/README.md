# Data

Two corpora are used. LonXplain is obtained from its authors; the conversation corpus is private and only its
expert-annotated phases are released, de-identified.

## LonXplain (Stages 1 and 2)

[LonXplain](https://doi.org/10.1007/978-3-031-35320-8_27) (Garg et al., 2023) is 3,522 Reddit posts labeled for
loneliness, with expert-annotated evidence spans for the lonely posts. Request it from the authors and place the file
at

```
data/lonxplain/raw/Lonesomenessdataset.csv
```

Then build the study's tables (CPU):

```bash
uv run python -m src.utils.data_prep           # -> data/lonxplain/posts.csv, sentences.json
```

`posts.csv` has one row per post (`id` = `lx0000`..., `text`, `label`, `fold`, the expert span as character offsets);
the ten stratified folds use seed 42 (`configs/default.yaml`) and folds 0, 2, 4, 6 and 8 form the shared test set of
1,761 posts. The 499 posts used for the evidence metrics are the held-out positive posts with an expert span, up to 100
per fold.

## Older-adult chatbot conversations (Stage 3)

Voice sessions between community-dwelling older adults and a wellbeing chatbot: 16 participants, 383 sessions, two
deployments (`deployment1`, `deployment2`; `deployment2b` is a later incremental export of the second). The raw
transcripts are private and are not in this repository. The pipeline that cleans, screens and formats them is included
(`src/stage3/`), and expects one CSV per deployment at `data/conversation/<deployment>.csv` with the columns
`nickname, conversation_id, timestamp, source, type, content`.

### Released: the expert annotations

`data/conversation/annotations/` holds the 34 phases that Claude Opus 5 flagged as possibly expressing loneliness and
that two experts annotated independently (7 participants):

| File | What |
|---|---|
| `expert1.docx`, `expert2.docx` | the filled-in annotation sheets: for each phase, the cleaned transcript, the checkbox "Expresses loneliness" and the annotator's quoted evidence |
| `annotations.csv` | one row per phase: `phase_id`, `corpus`, `conversation_id`, `type` (session phase), `nickname` (study pseudonym), `date`, `n_user_words`, `text` (the older adult's turns, as the detectors score them), then for each annotator `label` (1/0), `explanation` (verbatim), `span_chars` (character spans of the quoted evidence located in `text`), `n_spans`, `unlocated_quotes`, and `agreed` |
| `annotator_agreement.csv` | percent agreement (0.647), Cohen's kappa, Gwet's AC1, word-level span Jaccard (0.330) and F1 |

Expert 1 marked 32 of 34 phases lonely, expert 2 marked 22; they agree on 22. Names are replaced by `[NAME]` and
places by `[PLACE]`. The annotation rubric is Appendix C of the paper; the screening prompt is `screening_prompt` in
`configs/prompts.yaml`.

To rebuild `annotations.csv` from the sheets (CPU; needs the screened phases, i.e. the private corpus):

```bash
uv run python -m src.stage3.annotator_labels \
    --annotator data/conversation/annotations/expert1.docx \
    --annotator data/conversation/annotations/expert2.docx
```
