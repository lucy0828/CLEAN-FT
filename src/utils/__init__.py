"""Shared layer for every stage.

data_prep is the independent raw-data pipeline (posts, sentences, activation
cache) and the loaders Stages 1 and 2 read the processed data back with;
common holds paths, config, the method registry and metrics; sae_features,
dla and faithfulness carry the Stage 2 audit machinery.
"""
