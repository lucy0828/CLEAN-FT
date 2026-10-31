# The Stage 1 detection metrics

Stage 1 asks whether a method can detect loneliness at all. That is two
questions, and `results/detection.csv` keeps them apart:

- **Can it rank?** — does the score column sort present above absent, at any
  threshold at all? (`auroc`)
- **Can it decide?** — at the one threshold it actually fitted, how good is
  the yes/no call, and *which way does it err*? (`accuracy`, `fpr`, `fnr`)

A method can rank well and decide badly, if its threshold sits in the wrong
place. Keeping the columns side by side is what makes that visible.

| column | question | needs a threshold | chance |
|---|---|---|---|
| `auroc` | does the ranking separate the classes | no | 0.5 |
| `accuracy` | how often is the call right, either way round | yes | 0.547 on this corpus |
| `fpr_at_tpr90` | to catch 90% of present posts, how many absent posts does it drag in | no | 0.9 |
| `fpr` | how often does it flag a post that is absent | yes | 0.0 |
| `fnr` | how often does it miss a post that is present | yes | 0.0 |

Every trained method fills every column. A probe emits a score and no
decision of its own, so `src/stage1/train.py` fits it the same threshold the
fine-tuning arms get -- `best_f1_threshold` on that fold's *training* posts,
applied to the held fold -- and writes the resulting call beside the score.
The cut is therefore out of fold for every row in the table, and the decision
columns compare like with like across probes and arms.

A blank cell means the method cannot produce that number, never that it scored
zero. A probe row with empty decision cells is not threshold-free -- it is a
score file written before probes carried a threshold, and it needs a refit.

---

## 0. The shared test set — `train_post` and `test_post`

**Every row is scored on the same 1,761 posts.** They are the union of the
five study folds `src/stage1/train.py` holds out (`SHARED_FOLDS = 0 2 4 6 8`),
and `src/stage1/detection.py` restricts every method's scores to them before
computing anything. A method missing any of those posts is refused outright,
not published on whatever subset it happens to have.

This is not bookkeeping. Before it was enforced the table compared numbers
measured on three different corpus slices: probes, the fine-tuning arms and
mentalbert on these five folds; `glove_bigru` and `prompting` on all ten; and
the retired `anchor` cell on fold 0 alone. Fold 0 is the easiest fold in the
corpus -- `full_ft` reads 0.9395 there against 0.9265 pooled -- so the anchor's
apparent 1.6-point lead over `full_ft` was 0.3 points of method and 1.3 points
of evaluation set. That row is gone; `full_ft` is the same full fine-tune run
over all five folds.

- **`test_post`** is 1,761 in every row, by construction. It is published
  anyway, as the assertion that the restriction ran.
- **`train_post`** is 3,170: the posts each fold's model was fitted on, mean
  over the five folds (fold 0 has 3,169, the rest 3,170). `prompting` reads 0
  -- it is zero-shot and trains on nothing.

**What `train_post` is not.** These are cross-validation folds, so the five
training pools overlap and their union is the whole corpus. No post is ever
scored by a model that trained on it -- that is what out-of-fold means, and it
is the guarantee the table needs -- but `train_post` and `test_post` do not
name two disjoint sets of posts. Globally disjoint train and test sets would
take a single fixed split instead of cross-validation, at the cost of the
averaging and a much wider CI on every row.

---

## The construct is asymmetric, so the error columns are not interchangeable

The four annotation rules (`configs/prompts.yaml`, hash-pinned) are not
symmetric about the two error types. Three of them exist to stop a method
calling *present* on text that merely looks lonely, and one exists to stop it
calling *absent* on text where the void is real but old:

| rule | what it forbids | error it prevents | column |
|---|---|---|---|
| 2. Appraisal | reading a breakup, a divorce, living alone as loneliness without an explicit negative relational appraisal | false positive | `fpr` |
| 3. Interpersonal conflict exclusion | reading family arguments and fights as loneliness | false positive | `fpr` |
| 4. General distress exclusion | reading insomnia, exhaustion, depression as loneliness | false positive | `fpr` |
| 1. Temporal | discarding a past or lingering relational void because the post also carries positive affect | false negative | `fnr` |

Three of the four rules are false-positive rules. **`fpr` is therefore the
column this construct cares about most, and `fnr` is the guardrail that stops
the cheap way of satisfying it** — a method that never says present scores a
perfect `fpr` and violates the temporal rule on every post. `accuracy` folds
the two together, but at a weight -- one error, one unit -- that these rules do
not endorse, which is why it is published with both errors unfolded beside it.

No rule maps onto `auroc`. The rules govern where the line is drawn; AUROC is
the one column that refuses to draw one.

---

## 1. `auroc` — can it rank

**Is a present post scored above an absent post?**

The probability that a randomly drawn present post outranks a randomly drawn
absent post, integrated over every threshold:

    auroc = P(score(present) > score(absent))

Published as a single AUROC over the shared test set's pooled out-of-fold post
scores — never a mean of per-fold AUROCs — with a 2,000-draw post-level
bootstrap 95% CI in `auroc_lo` / `auroc_hi`. Because every row is scored on the
same 1,761 posts (§0), two CIs that overlap are overlapping on the same sample,
and a difference between two rows is a difference of method.

**Reading it.** Chance is 0.5. Invariant to class balance and to any monotonic
rescaling of the scores, so it is comparable across methods whose scores live
on different scales. Because it is threshold-free, a bad `auroc` cannot be
fixed by re-thresholding and a good one is no promise that the fitted
threshold is any good — which is why the decision cells are published beside
it. What `auroc` cannot tell you is *where on the curve* a method is cheap:
two methods with the same AUROC can charge very different numbers of false
alarms at the sensitivity a screening deployment would actually pick. That is
`fpr_at_tpr90` (§5).

---

## 2. Why there is no `f1`

`f1` was published here and has been withdrawn. Two properties made it the
wrong column to judge this study by:

1. **It charges a false positive and a false negative the same.** `fp` and
   `fn` carry identical weight in `2·tp / (2·tp + fp + fn)`. `glove_bigru` and
   `prompting` on gemma-2-2b scored 0.773 and 0.771 — indistinguishable —
   while their `fpr` differs by a factor of 2.4 (0.230 against 0.555). One of
   them obeys the exclusion rules and one does not, and `f1` cannot tell you
   which. Three of the four annotation rules are false-positive rules; a
   symmetric summary is the one thing this construct cannot use.
2. **It moves with prevalence.** This corpus is 54.7% present, so a degenerate
   method that answers "present" to every post scores **f1 = 0.707**. Read
   against a 0.5 chance line that looks like a passing grade; read against its
   actual floor, `glove_bigru`'s 0.773 was 0.065 of headroom, not 0.273.

`accuracy`, `fpr` and `fnr` are what `f1` was compressing, and
they are published unfolded. The threshold each row decides at is still
`best_f1_threshold`, fitted on the training folds (§6) — the objective is
unchanged, only the reporting is.

---

## 3. `fpr` — the exclusion rules

**Of the posts that are truly absent, what share did the method flag anyway?**

    fpr = fp / (fp + tn)

**Reading it.** This is the appraisal rule and the two exclusion rules,
measured. A method with a high `fpr` is, by construction, calling present on
posts a psychiatrist marked absent — and rules 2, 3 and 4 describe exactly the
posts where that is tempting: the breakup, the argument, the exhaustion.

Conditioned on the true class, so it does **not** move when the mix of present
and absent posts changes. That is what makes it the column to carry from this
54.7%-present corpus to a screening population, where `accuracy` would have
to be recomputed.

It is also the only own-threshold column that reads the true-negative cell at
all; `fnr` cannot see it, and `accuracy` sees it only mixed with `tp`.

**What it cannot do is rank methods.** `fpr` is read at each method's own
fitted threshold, and those thresholds are not the same place on the curve.
Compare rows on `fpr_at_tpr90`, which is this same rate at a matched 90%
sensitivity, and read `fpr` as the cost of the operating point this particular
method actually shipped.

**`fpr` IS the metric computed on the non-lonely posts**, and no subset gets
you a second one. Restricted to the absent posts there are no positives by
construction, so `tp` and `fn` are both zero: `fnr` is 0/0, and accuracy on
that subset is `tn/(tn + fp)` = exactly `1 - fpr`. "How good is this method at
not calling loneliness on a post that is not lonely" is one number, and it is
this column.

---

## 4. `fnr` — the temporal rule, and the guard on `fpr`

**Of the posts that are truly present, what share did the method miss?**

    fnr = fn / (fn + tp) = 1 - recall

**Reading it.** The direct measure of the temporal rule: a post whose
relational void is historical, or is voiced alongside a celebration, is still
present, and a method that discards it lands here. A false negative is a post
where **loneliness IS present and the method said absent** — not to be
confused with a true negative, which is a truly absent post the method
correctly left alone. In `true`/`false` `positive`/`negative`, the noun is
what the method predicted and the adjective is whether it was right.

`fnr` is published beside `fpr`, and both are errors reading
lower-is-better, because **`fpr` alone is half a metric**. Either one is
trivially optimised by a constant answer: a method that never says present
scores a perfect 0.0 `fpr` and an `fnr` of 1.0, and violates the temporal rule
on every post. A method is doing well when both are low, and the shape of the
trade between them is a property of the fitted threshold, not of the method —
which is what §5 exists to control for.

Neither error is drawn in any figure. `plot/plot_detection.py` plots `auroc`
and nothing else, because a bar chart reads as a verdict and every column here
except `auroc` is read at a threshold the method chose for itself. Read the
pair off the table, together, with `fpr_at_tpr90` beside them.

---

## 5. `accuracy` — the column the table is sorted on, and its two catches

**Of all 1,761 posts, the share called correctly, either way round.**

    accuracy = (tp + tn) / n

Rows in `results/detection.csv` are sorted by this column. It is the one
number a reader asks for first, and unfolding it into `fpr` and `fnr` beside
it is what keeps the sort honest. Two catches, both of which apply to the
sort and not just to the cell:

**1. It is read at each method's own fitted threshold.** Every method cut its
scores where `best_f1_threshold` landed on the training folds, and those cuts
are not the same place on each method's ROC curve. So the order here ranks
methods *and* the accidents of their calibration together. The case is in the
table: `mlp_probe` on gemma-2-2b has the lowest `fpr` in the study, 0.132
against `full_ft`'s 0.201, and buys it with an `fnr` of 0.345 against 0.127 —
it misses more than a third of the genuinely lonely posts. Held to one
sensitivity instead, `full_ft` pays 0.230 in false positives and `mlp_probe`
0.404 (`fpr_at_tpr90`). The probe was never more
faithful to the exclusion rules; it was quieter, and `accuracy` charges it
0.751 against `full_ft`'s 0.840 for it.

**2. Its floor is 0.547, not 0.5.** Unlike `fpr` and `fnr`, accuracy is not
conditioned on the true class, so it moves with the mix. This corpus is 54.7%
present, so a method that answers "present" to every post scores **0.547**,
and the same method carried to a 10%-prevalence screening population would
read something else entirely. Read the spread against that floor:
`attention_probe`'s 0.849 is 0.302 of headroom, not 0.349, and
`prompting`'s 0.695 is 0.148 — barely a quarter of the way from the
constant-answer baseline to the top of the table.

**What to rank on instead, and when.** `auroc` is threshold-free and
prevalence-invariant, so it is the column that survives being carried to
another population, and it is the one the frozen-vs-fine-tuned claim rests on.
The sort order here and the `auroc` order disagree — `attention_probe` and
`full_ft` swap, and `mlp_probe` falls seven places — and where they disagree,
the cause is a threshold, not a method.

## 6. What no column here measures

**Rule-conditional error.** The three false-positive rules do not describe all
absent posts — they describe a *subpopulation* of them: the post with an
interpersonal event and no appraisal, the post that is a family argument, the
post that is depression without a void. `fpr` averages those hard negatives
together with posts that are trivially, obviously absent. A method could
publish a respectable `fpr` and still fail every exclusion rule, if its errors
concentrate in that subset.

The rule-faithful measurement is `fpr` restricted to rule-relevant negatives,
and **the corpus cannot express it**: `data/lonxplain/posts.csv` carries id,
text, label, fold and the expert span, and no rule annotation to slice on.

This was attempted and withdrawn. A `false_positives.py` borrowed strata from
the `rules_triggered` field the Opus prompting pass emits -- one model's
account of which rules it thought it had applied -- and the attempt failed on
its own terms: that field marked the depression exclusion on 736 of the 797
absent posts, so the "stratum" was the whole negative class wearing a label.
A model's self-report of its own reasoning is not an annotation, and no number
conditioned on it belongs in this study. The scripts and their table are
quarantined in ~/lonxplain_removed_figs_20260827.

What would settle it is a rule annotation over the absent posts by someone
qualified to make one. Failing that, hardness can be defined from the data
rather than borrowed: an absent post that most methods false-positive on is
empirically hard, and per-method `fpr` on that consensus-hard subset -- scored
leave-one-out, so no method helps define the set it is graded on -- needs no
labels at all.

The one partial view available is the `prompting` arm, which self-reports
which rules it applied (`rules_triggered`, `src/baselines/prompting.py`).
Conditioning its errors on those flags suggests the rules are not acting as
constraints. On the shared test set's 797 absent posts, gemma-2-2b's `fpr` is
0.796 where it claims the appraisal rule against 0.101 where it does not, and
0.922 where it claims the temporal rule against 0.415 where it does not;
llama31-8b shows the same direction at half the level (0.408 vs 0.014, and
0.611 vs 0.188). The flags track *reasons to say present*, not gates that
withhold it. The depression exclusion is the one that behaves like a gate on
llama31-8b — 0.258 where claimed against 0.667 where not — and barely at all
on gemma-2-2b (0.488 against 0.585). Self-reported, unverified against any
gold rule label, and therefore suggestive only.

**Whether the threshold is the right one.** Every method fits its threshold
with `best_f1_threshold` on the training folds (`training_threshold` and
`_score_detection`, `src/stage1/train.py`), then applies it to the held fold.
For a probe the cut is measured on `post_scores`, the same quantity the
held-out scores are on, so it transfers with no rescaling step. That is a clean
out-of-fold procedure fitted to a symmetric objective — F1 weighs the two
errors equally, and this construct does not. `accuracy`, `fpr` and `fnr`
report where that objective landed; they do not claim it was the right
objective to pick. **Every decision column in this table, the one it is
sorted on included, inherits that threshold.** `fpr_at_tpr90` is the one that
does not.

---

## Reading the table

- **Empty `accuracy` / `fpr` / `fnr`.** The score file predates probe
  thresholds and carries no `pred` column. Re-run `src.stage1.train` for that
  method, then `src.stage1.detection`; nothing is re-thresholded at read time,
  where the only labels in reach are the held-out ones.
- **`test_post` is 1,761 in every row**, and a row that could not fill it does
  not exist: `src/stage1/detection.py` refuses a method missing any
  shared-test post rather than scoring it on a smaller sample (§0). There is
  no longer an `n_posts` column to compare across rows, because there is
  nothing left to compare.
- **No API-backbone row.** `--model opus` returns a label and no token
  distribution, so it has no `auroc` to set beside the decision columns the
  table is sorted on. A decision-only row cannot be set beside ranked ones, so
  `src/stage1/detection.py` refuses the model outright. The raw completions stay on disk at
  `results/detection/detect_prompt_opus.csv`.
- **Unparsed predictions.** A `prompting` completion whose label does not parse
  (35 of gemma-2-2b's 1,761; none of llama31-8b's) is counted as a hard
  "absent" call in `accuracy`, `fpr` and `fnr`: a method that returned no
  usable answer has still failed to flag the post, and dropping those rows
  would quietly score the arm on an easier sample than everyone else.
  `auroc` needs a number to rank and so uses the 1,727 posts that have one.
