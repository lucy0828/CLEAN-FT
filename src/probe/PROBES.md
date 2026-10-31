# The probing methods

Nine frozen-backbone readouts live in `src/probe/`. None of them moves a
backbone weight: every one reads cached activations and fits only how to read
them. `linear_probe` is the reference point — **each of the others changes
exactly one thing** (pooling, capacity, basis, fitting procedure, or
preprocessing), which is what makes the table below readable as a set of
controlled comparisons rather than a list of models. Two cells change two
things on purpose, each the interaction of a pair: `attention_mlp_probe`
(pooling × capacity) and `sae_attention_probe` (pooling × basis).

Layers are quoted for `gemma-2-2b` as **cached layer indices**; `llama31-8b`
uses the same rules at its own depths (`DENSE_PROBE_LAYER`, `SAE_PROBE_LAYER`
in `src/utils/common.py`). Read the note under the table before comparing the
L17 and L18 cells — they are one block apart, and the SAE's own layer number
is not its cached index.

| method key | display | basis / layer | pooling | readout fitted | token decomposition | what it isolates |
|---|---|---|---|---|---|---|
| `linear_probe` | Linear | raw residual, L17 (0.65 depth) | **mean** | L2 logistic, `C=1.0` | **exact + intrinsic** — post score is the token-count-weighted mean of word scores | *(the reference point)* |
| `attention_probe` | Attn-pool | raw residual, L17 | **learned** `softmax(Xq)` | `w`, `q`, `b` — 500 steps @ `1e-3` | exact but **contextual** — `alpha_t` is a softmax over the whole post | does learned aggregation beat uniform? |
| `mlp_probe` | MLP | raw residual, L17 | mean | MLP(256), `alpha=1e-4`, IG 256 steps | integrated-gradient direction, redistributed by the pooling identity — additive, but in context | does nonlinearity close a linear gap? |
| `attention_mlp_probe` | Attn-pool MLP | raw residual, L17 | **learned** `softmax(Xq)` | `q` + MLP(256) jointly, 500 steps @ `1e-3` | **inexact** — IG direction redistributed by `alpha_t`; closes by completeness, not by identity | does capacity still buy anything once the pooling is learned? |
| `sae_probe` | SAE | **Gemma Scope 16k latents**, SAE layer 17 = cached L18 | mean | L1 logistic | **exact + intrinsic** under mean pooling | does the model's own basis beat raw activations? |
| `sae_attention_probe` | SAE attn-pool | same dictionary, same site | **learned** `softmax(Zq)` in the SAE basis | `q`, `w`, `b` — 500 steps @ `1e-3`, L1 prox on `w` | exact but **contextual** — `alpha_t (w . z_t)` | does learned pooling still buy on the dictionary? |
| `afr_probe` | AFR | raw residual, L17 | mean | `linear_probe` **+ reweighted last-layer retrain** | exact (the composite is affine) | shortcut robustness without group labels |
| `shrunk_attention_probe` | Attn-pool (shrunk q) | raw residual, L17 | **learned** `softmax(Xq)` | `w`, `q`, `b` + **L2 on `q`**, penalty chosen on inner folds | exact but contextual, as `attention_probe` | is the pooling gain real, or the query overfitting? |

## The query was the only unpenalized object, and that is testable

`w` is penalized in every arm that has a penalty — L2 in `linear_probe`, L1 in
both SAE arms. `q` never was. It is `hidden` free parameters fitted by
unconstrained Adam, braked only by a small init and a fixed 500 steps. Order
every pooling contrast in the study by the query's sample-to-parameter ratio
(n = 3,170 training posts):

| basis | d(q) | n/d | ΔAUROC from learned pooling | Δ word AUPRC |
|---|---|---|---|---|
| gemma raw | 2,304 | 1.38 | **+0.093** | +0.085 |
| llama raw | 4,096 | 0.77 | +0.070 | +0.016 |
| gemma SAE | 16,384 | 0.19 | +0.019 | +0.052 |
| llama SAE | 32,768 | 0.097 | **−0.075** | −0.071 |

ΔAUROC is monotone in n/d across two backbones and two bases. Δ word AUPRC is
monotone with one inversion (gemma-SAE above llama-raw). Read that way, the
`sae_attention_probe` sign flip on llama is not a fact about dictionaries or
about layer choice — it is the same curve, further along, and the 2026-08-27
diagnostic agrees: penalising the query recovers llama to 0.9233 against
`sae_probe`'s 0.9256.

One arm tests it, by shrinking the query toward zero.

**`shrunk_attention_probe`** keeps the free query and adds `q_l2 * ||q||^2`. As
the penalty grows the query goes to zero, the softmax goes uniform, and the arm
becomes `linear_probe`; at zero it *is* `attention_probe`, bit for bit. The
penalty therefore interpolates the two ends of the pooling contrast the study
already reports, rather than adding a third thing.


### The penalty is selected on inner folds, and that is not a loophole

`configs/default.yaml` refuses tuning, and the 2026-08-27 `q_l2 = 3.0`
diagnostic was picked by looking at the held fold, which is exactly what it
refuses. `attention_probe.select_q_l2` splits the **training pool** by the
corpus's own stratified fold column and scores candidates on folds inside it;
the outer held fold is not in the pool and cannot be reached. The grid is fixed
in the config, the inner folds are the first three by id, and ties break toward
the larger penalty — so the selection is a deterministic function of (outer
fold, seed) and adds no new randomness. The selected value rides on every
scored row as `q_l2_selected`, because how hard the query had to be held down
is the finding rather than an implementation detail.

`attention_probe` itself is untouched: its published rows reproduce exactly,
which is enforced rather than asserted — the free-query path is byte-identical
to the pre-2026-08-28 solver at `q_l2 = 0`.

### The SAE is at layer 17. Its cache index is 18.

These are the same site named two ways, and the off-by-one is the single
easiest thing to "fix" into a silent bug.

The activation cache is written from HuggingFace `output_hidden_states`, where
`hidden_states[k]` is the *input* to block k — equivalently the output of block
k-1. The pinned Gemma dictionary
(`layer_17/width_16k/canonical`, `configs/default.yaml`) was trained at
`blocks.17.hook_resid_post`, which is the input to block 18:

| what | value |
|---|---|
| SAE's own layer (`sae.gemma-2-2b.layer`) | **17** |
| hook it was trained on | `blocks.17.hook_resid_post` |
| cached layer that holds that site | **18** = 17 + `SAE_CACHE_OFFSET` |
| `SAE_PROBE_LAYER["gemma-2-2b"]` | **18** |

So `SAE_PROBE_LAYER == 18` is what makes the probe read layer 17. Setting it to
17 would evaluate the dictionary against `blocks.16.hook_resid_post`, one block
early. Nothing raises — the shapes match and the latents are simply wrong.

**The dense rows and the SAE row are not at the same site.** The dense probes
read cached L17, which is `blocks.16.hook_resid_post`; the SAE reads cached
L18, which is `blocks.17.hook_resid_post`. The two differ by one block, because
the dense layer is chosen by depth fraction (0.65) and the SAE layer is
dictated by which Gemma Scope dictionary supports the construct. That is worth
knowing when reading the "does the model's own basis beat raw activations?"
comparison: the basis changes and the site moves one block at the same time.

`reconstruction_diagnostics` in `src/probe/sae_probe.py` settles the mapping
empirically — a dictionary reconstructs well at its own site and poorly one
block away. Re-run it across cached 17 and 18 after any repin.

## The pooling × capacity factorial

Two of the changes above vary independently, so the four combinations form a
2×2 and the interaction is measurable rather than assumed. AUROC on the shared
1,761-post test set (`results/detection.csv`):

**gemma-2-2b**

| | linear head | MLP(256) head | capacity buys |
|---|---|---|---|
| **mean pooling** | `linear_probe` 0.837 | `mlp_probe` 0.872 | **+0.035** [+0.024, +0.046] |
| **learned pooling** | `attention_probe` **0.930** | `attention_mlp_probe` 0.924 | **−0.005** [−0.009, −0.001] |
| **pooling buys** | +0.092 [+0.078, +0.108] | +0.052 [+0.041, +0.064] | |

**llama31-8b**

| | linear head | MLP(256) head | capacity buys |
|---|---|---|---|
| **mean pooling** | `linear_probe` 0.835 | `mlp_probe` 0.868 | **+0.034** [+0.024, +0.043] |
| **learned pooling** | `attention_probe` **0.905** | `attention_mlp_probe` 0.885 | **−0.020** [−0.028, −0.012] |
| **pooling buys** | +0.070 [+0.057, +0.084] | +0.017 [+0.010, +0.025] | |

Paired bootstrap, 2,000 draws, same posts for both arms of every comparison
(`plot/plot_detection.py --paired` uses the same machinery). Every interval
excludes zero.

**Read it as a crossover, and it replicates.** A nonlinear head is worth
~+0.034 AUROC on top of mean pooling on both backbones, and is worth less than
nothing on top of a learned query on both — −0.005 on gemma, −0.020 on llama.
The two factors are substitutes: they fix the same failure, and the MLP head
is the weaker fix. That is the empirical content of "pooling, not capacity" —
the claim is not that capacity never helps, it is that capacity helps *only
while the pooling is bad*.

The same reading from the other margin: learned pooling is worth +0.092 /
+0.070 on a linear head but only +0.052 / +0.017 once an MLP head is already
there. Whichever factor arrives second finds less left to fix.

Two consequences for the study:

1. **The linear head is kept on evidence, not on preference.** It is the only
   one of the two with an exact additive decomposition, and the factorial says
   it costs nothing to keep.
2. **`attention_mlp_probe` is a diagnostic, not a candidate readout.** It
   exists to fill this cell. Its token attribution closes by
   integrated-gradient completeness rather than by identity, so it cannot
   carry Stage 2's own-decomposition audit — see the next section.

`sae_probe` is *not* a cell in this table. It changes the **basis**, a third
axis, and keeps mean pooling and a linear head:

| | basis | pooling | head |
|---|---|---|---|
| `linear_probe` | raw residual | mean | linear (L2) |
| `sae_probe` | SAE latents | mean | linear (L1) |

so the controlled basis comparison is `sae_probe` against `linear_probe`:
0.870 vs 0.837 on gemma, 0.898 vs 0.835 on llama. Note that comparison also
swaps L2 for L1 and moves one block (see the layer note above), so it is not
as clean as the 2×2. `sae_attention_probe` fills the attention-pooled cell of
the dictionary basis, so the pooling contrast is measured on the SAE too.

**Max pooling is not part of the design.** A max-pooled arm of each readout was
run and removed on 2026-08-27 (quarantined under
`~/lonxplain_removed_maxpool_20260827`). Its winner-take-all attribution
credits each coordinate to the one token that maximised it, so most words score
exactly zero and a word ranking is mostly ties — which depresses average
precision for a reason that is a property of the metric under ties rather than
of the evidence the probe found. The pooling axis the study reports is
therefore uniform against learned.

## Why the decomposition column is the important one

Stage 2 scores word evidence against the psychiatrist's spans, so a method is
only usable there if its word scores *are* its decision rather than a post-hoc
explanation of it. Under mean pooling and a linear readout that identity is
exact:

    score(post) = w . mean_t(X_t) + b = mean_t(w . X_t) + b

so the post score is the token-count-weighted mean of the word scores and the
two cannot disagree. It needs **both** properties, which is why the two arms
that give one up are the interesting controls:

- `attention_probe` keeps additivity (`score - b = sum_t alpha_t (w . X_t)`)
  but loses intrinsicness: `alpha` is a softmax over the whole post, so a
  token's contribution depends on what else the post says. Faithfulness there
  has to be measured by erasure, not asserted by construction.
- `mlp_probe` gives up intrinsicness for a second reason. Its logit is not a
  projection of the pooled vector at all, so there is no fixed `w`; what there
  is exactly is the integrated-gradient direction `gbar` between the average
  training post and this post, for which completeness gives
  `logit(post) - logit(baseline) = gbar . (xbar - baseline)`. `gbar` is a
  single direction, so the pooling identity carries it to tokens unchanged.
  The decomposition is additive and closes to a measured residual (asserted
  per fold), but like the attention probe's it is a contribution *in context*:
  the direction is refitted per post. This is the one place the study computes
  an attribution rather than reading one off the head, and the figures mark
  it.

## Nothing is tuned

The layer is fixed a priori at 65% of depth, `C` is fixed at 1.0, and the SAE
dictionary is pinned in `configs/default.yaml` and hashed into
`results/decisions.json`. A probe that picks its layer on
labels is quietly spending the annotation budget it claims to save.

Two honest caveats the modules state about themselves:

- `afr_probe` deviates from published AFR, whose base is normally an ERM
  *fine-tuned* network; here the base is a frozen-feature probe, so an AFR row
  measures AFR's reweighting on frozen features. Its `gamma` is not tuned
  (published AFR selects it on a group-labelled validation set).
- `sae_probe` treats "sparse" as a measured claim, not a penalty: the fitted
  non-zero coefficient count is written on every output row, because an L1
  probe with 14,000 of 16,384 coefficients live is a dense probe wearing an L1
  penalty.
- `attention_mlp_probe`'s hidden width is 256 because that is `mlp_probe`'s,
  so the capacity axis of the 2×2 is the same size in both rows. It is
  *matched*, not *tuned*, and no other width was tried. The negative capacity
  effect replicates on both backbones and its interval clears zero on both, so
  the direction is not in doubt; what a width sweep could still change is the
  magnitude, which is small on gemma (−0.005) and not on llama (−0.020).

## Naming

`attention_mlp_probe` is named for both things it changes, in the order the
2×2 reads them: pooling first, then the head. It is the one module whose name
carries two factors, because it is the one module that varies two.

`linear_probe` was called `mean_pool_logistic` until 2026-08-26. It was renamed
for two reasons: every sibling ends in `_probe`, and mean pooling is the family
*default* (`mlp_probe`, `sae_probe` and `afr_probe` all mean-pool too), so
naming the pooling named the thing that is shared rather than the thing that is
distinct. "Linear" is what separates it from `mlp_probe`, and it reads correctly
in the derived method — AFR is the linear probe plus reweighting.

The published prose in `PAPER.md` and `RESULTS.md` still says "mean-pool
logistic" where it is drawing the pooling contrast against "attention-pool
logistic", which is deliberate: there the parallel wording is the point.
