"""SAE plumbing: the pinned dictionaries, the token alignment the word-level
metrics are scored over, and the hooks that read a running classifier.
Not run directly.

No external clinical feature list is involved anywhere in this module. The
frozen-manifest layer was removed on 2026-08-26 (see
~/lonxplain_removed_sae_metrics_20260826): blind re-inspection showed the
pinned features were largely lexical and register detectors rather than the
loneliness construct, so both metrics that depended on them were retired.
Every surviving readout ranks by the method's OWN attribution.

1. DICTIONARIES — load the pinned Gemma Scope / Llama Scope SAE.
2. ALIGNMENT — map post tokens to psychiatrist spans and to natural-language
   words, the unit `dla_auprc` scores word AUPRC over.
3. HOOK ADAPTER — TransformerLens HookPoints at the decoder-block residual
   outputs of the exact classifier path, byte-for-byte no-op when unhooked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from transformer_lens import HookedRootModule
from transformer_lens.hook_points import HookPoint

from src.utils.common import SAE_CACHE_OFFSET, sae_cache_layer
from src.utils.data_prep import build_prompt, words

# ========================================================= 2. SAE profiling


@dataclass(frozen=True)
class TokenAlignment:
    """Post-content token indices and their gold/word alignment."""

    token_indices: np.ndarray
    word_indices: np.ndarray
    gold_mask: np.ndarray
    n_words: int
    truncated: bool


@dataclass(frozen=True)
class SAEProfile:
    logits: torch.Tensor
    residual: torch.Tensor
    feature_activations: torch.Tensor
    reconstruction_rel_l2: torch.Tensor
    mean_active_features: torch.Tensor


def load_pretrained_sae(
    sae_config: dict,
    *,
    device: str,
    dtype: str = "float32",
):
    """Load exactly the release/id pinned in config.

    This function downloads model artifacts when absent. The CLI safety gate
    must therefore be checked by the caller before invoking it.
    """
    release = sae_config.get("release")
    sae_id = sae_config.get("sae_id")
    if not release or not sae_id:
        raise ValueError("SAE release and sae_id must be resolved and pinned")
    from sae_lens import SAE

    return SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
    )


def validate_sae_compatibility(
    sae,
    *,
    hidden_size: int,
    expected_hook_layer: int | None = None,
) -> dict[str, Any]:
    """Fail early when a downloaded SAE does not match the hooked residual."""
    decoder = getattr(sae, "W_dec", None)
    if decoder is None or decoder.ndim != 2:
        raise ValueError("loaded SAE must expose a rank-2 W_dec")
    width, sae_input_size = (int(value) for value in decoder.shape)
    if sae_input_size != int(hidden_size):
        raise ValueError(
            f"SAE input width {sae_input_size} does not match model hidden "
            f"size {hidden_size}"
        )
    sae_cfg = getattr(sae, "cfg", None)
    metadata = getattr(sae_cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None)
    if hook_name is None:
        hook_name = getattr(sae_cfg, "hook_name", None)
    expected_hook = (f"blocks.{expected_hook_layer}.hook_resid_post"
                     if expected_hook_layer is not None else None)
    if expected_hook and hook_name and str(hook_name) != expected_hook:
        raise ValueError(
            f"SAE hook {hook_name!r} does not match expected {expected_hook!r}"
        )
    return {
        "sae_width": width,
        "sae_input_size": sae_input_size,
        "sae_hook_name": str(hook_name) if hook_name else expected_hook,
    }


@torch.no_grad()
def profile_with_sae(
    adapter: PooledClassifierHookAdapter,
    sae,
    *,
    layer: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    post_mask: torch.Tensor,
) -> SAEProfile:
    """Capture one residual stream and its SAE feature activations."""
    if layer < 0 or layer >= adapter.n_layers:
        raise ValueError(f"layer {layer} outside adapter depth {adapter.n_layers}")

    cache: dict[str, torch.Tensor] = {}

    def capture(residual: torch.Tensor, hook) -> torch.Tensor:
        cache["residual"] = residual.detach()
        return residual

    with adapter.hooks(fwd_hooks=[(adapter.hook_name(layer), capture)]):
        logits = adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            post_mask=post_mask,
        )
    if "residual" not in cache:
        raise AssertionError(f"residual HookPoint {layer} never fired")

    residual = cache["residual"]
    parameter = getattr(sae, "W_dec", None)
    if parameter is None:
        parameter = next(sae.parameters())
    encoded = sae.encode(
        residual.to(device=parameter.device, dtype=parameter.dtype)
    )
    reconstructed = sae.decode(encoded)
    residual_for_sae = residual.to(
        device=reconstructed.device, dtype=reconstructed.dtype
    )
    content = post_mask.to(
        device=reconstructed.device, dtype=reconstructed.dtype
    ).unsqueeze(-1)
    error = ((reconstructed - residual_for_sae) * content).pow(2).sum(
        dim=(1, 2)
    ).sqrt()
    scale = (residual_for_sae * content).pow(2).sum(dim=(1, 2)).sqrt()
    reconstruction_rel_l2 = error / scale.clamp_min(1e-12)
    token_count = content.squeeze(-1).sum(dim=1).clamp_min(1)
    mean_active_features = (
        ((encoded > 0).to(encoded.dtype) * content)
        .sum(dim=(1, 2))
        / token_count
    )
    return SAEProfile(
        logits=logits.detach(),
        residual=residual,
        feature_activations=encoded.detach(),
        reconstruction_rel_l2=reconstruction_rel_l2.detach(),
        mean_active_features=mean_active_features.detach(),
    )


def align_post_tokens(
    tokenizer,
    text: str,
    *,
    max_len: int,
    span_start: int | None = None,
    span_end: int | None = None,
) -> TokenAlignment:
    """Align chat-templated token offsets to post words and a gold char span."""
    full, post_start, post_end = build_prompt(tokenizer, text)
    encoded = tokenizer(
        full,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    untruncated = tokenizer(
        full,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    post_words = words(text)
    token_indices: list[int] = []
    word_indices: list[int] = []
    gold: list[bool] = []

    has_gold = (
        span_start is not None
        and span_end is not None
        and span_start >= 0
        and span_end > span_start
    )
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if not (start < post_end and end > post_start and end > start):
            continue
        char_start = max(0, start - post_start)
        char_end = min(len(text), end - post_start)
        word_index = next(
            (
                index
                for index, word in enumerate(post_words)
                if word["norm"]
                and word["start"] < char_end
                and word["end"] > char_start
            ),
            None,
        )
        if word_index is None:
            continue
        token_indices.append(token_index)
        word_indices.append(word_index)
        gold.append(
            bool(
                has_gold
                and char_start < int(span_end)
                and char_end > int(span_start)
            )
        )

    return TokenAlignment(
        token_indices=np.asarray(token_indices, dtype=np.int64),
        word_indices=np.asarray(word_indices, dtype=np.int64),
        gold_mask=np.asarray(gold, dtype=bool),
        n_words=len(post_words),
        truncated=len(untruncated) > len(encoded["input_ids"]),
    )


def token_plausibility_auprc(
    sequence_scores: np.ndarray | torch.Tensor,
    alignment: TokenAlignment,
) -> float:
    """AUPRC over aligned content tokens; undefined single-class rows are NaN."""
    if torch.is_tensor(sequence_scores):
        sequence_scores = sequence_scores.detach().float().cpu().numpy()
    sequence_scores = np.asarray(sequence_scores, dtype=float)
    if sequence_scores.ndim != 1:
        raise ValueError("sequence_scores must be one-dimensional")
    if alignment.token_indices.size == 0:
        return float("nan")
    if alignment.token_indices.max() >= len(sequence_scores):
        raise ValueError("token alignment exceeds score sequence length")
    scores = sequence_scores[alignment.token_indices]
    finite = np.isfinite(scores)
    labels = alignment.gold_mask
    if not finite.all() or len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels.astype(int), scores))


def aggregate_word_scores(
    sequence_scores: np.ndarray | torch.Tensor,
    alignment: TokenAlignment,
    *,
    reduction: str = "max",
) -> np.ndarray:
    """Aggregate aligned sub-token scores to natural-language words."""
    if reduction not in {"max", "mean", "sum"}:
        raise ValueError(f"unknown word score reduction {reduction!r}")
    if torch.is_tensor(sequence_scores):
        sequence_scores = sequence_scores.detach().float().cpu().numpy()
    sequence_scores = np.asarray(sequence_scores, dtype=float)
    out = np.full(alignment.n_words, np.nan, dtype=float)
    for word_index in range(alignment.n_words):
        token_positions = alignment.token_indices[
            alignment.word_indices == word_index
        ]
        if token_positions.size == 0:
            continue
        values = sequence_scores[token_positions]
        if reduction == "max":
            out[word_index] = float(np.max(values))
        elif reduction == "mean":
            out[word_index] = float(np.mean(values))
        else:
            out[word_index] = float(np.sum(values))
    return out


def sae_result_metadata(
    sae_config: dict,
    *,
    layer: int,
    n_tokens: int,
    truncated: bool,
) -> dict[str, Any]:
    """Stable metadata columns written on every SAE-site row."""
    return {
        "sae_layer": layer,
        "sae_release": sae_config.get("release"),
        "sae_id": sae_config.get("sae_id"),
        "neuronpedia_source": sae_config.get("neuronpedia_source"),
        "n_content_tokens": n_tokens,
        "truncated": bool(truncated),
        "audit_protocol": "own_feature_v1",
    }


# The cache off-by-one (SAE_CACHE_OFFSET) lives in src.utils.common, next to the
# probe layers that depend on it, so the SAE pin and the cached layer the probes
# read off it stay one fact. Re-exported here, where the audit code looks for it.


def cached_site_layer(sae_config: dict) -> int:
    """The cached layer the pinned dictionary actually reconstructs."""
    return sae_cache_layer(sae_config)


# ============================================================ hook adapter
# Formerly src/utils/sae_features.py §3: TransformerLens HookPoints at the
# decoder-block residual outputs of this exact classifier path, preserving
# the original forward byte-for-byte when no hook changes an activation.



def decoder_blocks(classifier: nn.Module) -> nn.ModuleList:
    """Return the decoder blocks whose outputs are ``resid_post`` tensors."""
    backbone = classifier.backbone
    for path in ("layers", "model.layers", "transformer.h", "encoder.layer"):
        obj = backbone
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj):
            return obj
    raise AssertionError(
        f"no decoder block list found on {type(backbone).__name__}; "
        "TransformerLens hooks cannot locate the residual stream"
    )


def hidden_and_rebuild(output):
    """Return ``(hidden, rebuild)`` for tensor or tuple block outputs."""
    if torch.is_tensor(output):
        return output, lambda hidden: hidden
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        rest = output[1:]
        return output[0], lambda hidden: (hidden,) + rest
    raise AssertionError(
        f"unexpected decoder block output {type(output)}; "
        "cannot locate residual stream"
    )


class PooledClassifierHookAdapter(HookedRootModule):
    """Run a pooled classifier with TransformerLens residual HookPoints.

    Hook names are ``hook_resid_post.<layer>``. Hooks receive and must return a
    tensor shaped ``[batch, tokens, hidden]``. PyTorch block handles exist only
    during ``forward`` and are removed in ``finally`` even if a hook fails.
    """

    classifier: nn.Module
    hook_resid_post: nn.ModuleList

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.hook_resid_post = nn.ModuleList(
            [HookPoint() for _ in decoder_blocks(classifier)]
        )
        self._active_block_handles = 0
        self.setup()

    @property
    def n_layers(self) -> int:
        return len(self.hook_resid_post)

    @staticmethod
    def hook_name(layer: int) -> str:
        return f"hook_resid_post.{layer}"

    @property
    def active_block_handles(self) -> int:
        """Number of temporary Hugging Face block hooks currently attached."""
        return self._active_block_handles

    def _block_hook(self, layer: int) -> Callable:
        point = self.hook_resid_post[layer]

        def apply_hook(module, args, output):
            hidden, rebuild = hidden_and_rebuild(output)
            return rebuild(point(hidden))

        return apply_hook

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        post_mask=None,
        inputs_embeds=None,
    ):
        blocks = decoder_blocks(self.classifier)
        handles = [
            block.register_forward_hook(self._block_hook(layer))
            for layer, block in enumerate(blocks)
        ]
        self._active_block_handles = len(handles)
        try:
            return self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                post_mask=post_mask,
                inputs_embeds=inputs_embeds,
            )
        finally:
            for handle in handles:
                handle.remove()
            self._active_block_handles = 0


@torch.no_grad()
def assert_noop_parity(
    adapter: PooledClassifierHookAdapter,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    post_mask: torch.Tensor,
    atol: float = 1e-5,
) -> float:
    """Assert that inactive HookPoints preserve the classifier's own logits."""
    expected = adapter.classifier(
        input_ids=input_ids,
        attention_mask=attention_mask,
        post_mask=post_mask,
    )
    actual = adapter(
        input_ids=input_ids,
        attention_mask=attention_mask,
        post_mask=post_mask,
    )
    difference = float((expected.float() - actual.float()).abs().max().cpu())
    if difference >= atol:
        raise AssertionError(
            f"TransformerLens no-op parity failed: max|diff|={difference:.3g} "
            f">= {atol:.3g}"
        )
    if adapter.active_block_handles:
        raise AssertionError("temporary decoder-block hooks leaked after forward")
    return difference
