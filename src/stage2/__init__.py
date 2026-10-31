"""Stage 2, one runnable CLI per audit metric, each writing one results CSV:
dla_auprc (plausibility: own-decomposition word AUPRC) and word_deletion
(faithfulness).

    dla_auprc       does a method's own word ranking agree with the
                    psychiatrist's spans?
    word_deletion   delete the top-k% words that method itself ranked highest,
                    re-encode, and measure the normalized AOPC against a
                    count-matched random-word control. Asks whether the
                    explanation is faithful to that method's own decision.

Plausibility says the ranking AGREES with the expert; word_deletion says the
ranking is what the model USED. Neither licenses the feature-distortion reading
alone, which is why both are published.

Every raw row carries `ranking_source`, which is now always `own_attribution`:
the shared-ranking companion `sae_word_deletion` was removed on 2026-08-28
(quarantine: ~/lonxplain_removed_saeworddeletion_20260828). It measured
construct dependence rather than faithfulness -- its own docstring said so --
and it never left a DRAFT manifest. The external clinical-SAE knockout metrics
were retired on 2026-08-26 and the SAE own-feature knockout with them;
activation patching was dropped from the study on 2026-08-26.

The equations and how to read each number are in `src/stage2/METRICS.md`.
"""
