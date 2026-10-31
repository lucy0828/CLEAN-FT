"""Stage 1: train.py trains every detector on the loneliness posts (probes,
fine-tuning arms, references); detection.py reports AUROC to
results/detection.csv. Data preparation lives in src/utils/data_prep.py.

What each published column computes, and which annotation rule it answers to,
is in `src/stage1/METRICS.md`.
"""
