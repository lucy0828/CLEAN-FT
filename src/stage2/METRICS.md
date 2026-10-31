# The Stage 2 audit metrics

Stage 1 asks whether a method can *detect* loneliness. Stage 2 asks what its
decision was made of, in two questions that are easy to conflate:

- **Plausibility** — are the words a method points at the words the
  psychiatrist underlined? (`dla_auprc`)
- **Faithfulness** — does the decision actually *depend* on the words it
  named? (`word_deletion`)

A method can be plausible and unfaithful (it names the right words but decides
on something else) or faithful and implausible (its decision really does hinge
on the words it names, and they are the wrong ones). That gap is the point of
running both.

A third metric, `sae_word_deletion`, ranked the words by a frozen clinical-SAE
manifest instead of by the method's own attribution and asked how much each
decision depended on the dictionary's vocabulary. It was removed on 2026-08-28
(quarantine: `~/lonxplain_removed_saeworddeletion_20260828`): it was not a
faithfulness measure by its own definition, it was off-target for the
distortion claim, its rerun separated nothing, and its manifest never left
DRAFT.

Neither column is **shortcut robustness**, and neither is **feature
distortion**. Distortion is what fine-tuning does to pretrained geometry
(Kumar et al., 2022). A shortcut is a non-construct cue the decision uses,
visible only when that cue is manipulated. A drop in word AUPRC is not
evidence that fine-tuning latched onto formatting; a frozen probe can take
such a cue without distorting anything. Detection AUROC being a tie is why
these questions are the study: we need the path (real, then clinical), not
another point of AUROC.

| CLI | question | ranks words by | intervention | output |
|---|---|---|---|---|
| `dla_auprc` | plausibility | the method's own contribution | none — ranking only | `results/dla_auprc.csv` |
| `word_deletion` | faithfulness to itself | the method's own contribution | delete from text | `results/word_deletion.csv` |

Both score the same preregistered subset: up to `audit.eval_posts_per_fold`
present posts per fold that carry an aligned expert span, drawn with a
fold-pinned seed (`span_eval_ids`), so the two tables describe the identical
posts.

---

## 1. `dla_auprc` — plausibility

**Does the method point where the psychiatrist pointed?**

Every method ranks tokens by its *own* additive contribution to its *own*
logit. Under mean pooling and a linear readout that contribution is not an
explanation of the decision, it **is** the decision:

    score(post) = w . mean_t(X_t) + b = mean_t(w . X_t) + b = mean_t(s_t) + b

so `s_t = w . X_t` are terms that reconstruct the post score exactly. Sub-word
tokens are pooled to words with the pinned reduction (`audit.
word_score_aggregation`, `max`):

    s_j = max over { s_t : token t overlaps word j }

Each post is then scored as a ranking problem against its gold span — every
word is positive if it falls inside the psychiatrist's span, negative
otherwise — and the published number is the mean over posts with a 2,000-draw
bootstrap CI:

    word AUPRC = average_precision(gold_j, s_j)

Token AUPRC is reported beside it, on the same rule at token granularity.

**Reading it.** Chance is the post's own base rate (the gold fraction), not
0.5, so an AUPRC near the base rate means "no better than pointing at random
words". Posts that are all-gold or no-gold have no ranking metric and are NaN
rather than counted. `mlp_probe` publishes no cell here at all: it has no
additive decomposition, and inventing one would be a post-hoc explanation
rather than its decision.

---

## 2. `word_deletion` — faithfulness to the method's own explanation

**If we take away the words the method says it used, does its decision move?**

Same ranking as `dla_auprc` — the method's own word scores. Now they are
deleted from the raw text and the post is re-encoded. For each K on the pinned
grid (`faithfulness.k_grid` = 1, 2, 5, 10, 20, 50 %):

    n_K   = clamp(ceil(n_active * K / 100), 1, n_active)    words deleted
    d(K)  = (logit_clean - logit_deleted(K)) / (logit_clean - logit_empty)

`d(K)` is the fraction of the model's whole clean-to-empty logit gap that the
deletion removed, so it is comparable across methods with different logit
scales. The curve is summarised by the trapezoidal area over the grid with the
clean point `(0, 0)` inserted, divided by the covered range:

    AOPC = trapezoid(d, K) / max(K)

Deleting words shortens the post, which is off-manifold as text regardless of
*which* words go. So the number published is never the raw AOPC but the
**margin** against a control that deletes `n_K` **random** words
(`faithfulness.random_draws` = 5 replicates, averaged):

    margin = AOPC(target) - AOPC(random control)

One AOPC per post per arm, differenced per post, then averaged over posts with
a 2,000-draw bootstrap CI. Posts whose clean-to-empty gap is not positive are
dropped on the probe path (their per-K statistic is undefined), and posts
flagged `small_logit_gap` are dropped everywhere.

**Reading it.** A positive margin means the method's own top words carry more
of its decision than an equal number of its other words. Near zero means the
explanation is decoration: the decision survives losing exactly the words the
method nominated.

**Cost.** Deletion changes the TEXT, so the activation cache cannot be reused —
every variant goes back through the backbone. That is the only reason the probe
half needs a GPU (`--compute --allow-model-process`); the fine-tuning arms'
cells are deposited by Stage 1 training itself.

---

## Caveats the metrics state about themselves

- **Deleted text is off-manifold as TEXT even though it is on-manifold as
  activations.** A shortened post is not a post the model was trained on. The
  count-matched random-word control pays that same cost, which is why the
  MARGIN and not the raw AOPC is what gets published.
- **An AOPC margin rewards any ranking correlated with word salience.** It is a
  ranking test, not a proof that the named words are the decision; a method
  whose pooling concentrates the decision on few tokens can score high for that
  reason alone. The pooling axis is drawn in `plot/plot_aopc.py` so that
  confound is visible rather than argued about.
- **`n` differs by method** because the published margin drops posts whose
  clean-to-empty logit gap is not positive — the denominator of
  `normalized_decay`, so the per-post statistic is undefined there, not merely
  noisy. Until 2026-08-28 that filter was applied to the probes and not to the
  fine-tuning arms, which depressed every arm's margin; it is now applied on
  both paths.
