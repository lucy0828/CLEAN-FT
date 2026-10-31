"""Fine-tune the backbone under the study's adaptation recipe.

Not run directly — src/stage1/train.py drives it.

This is the pipeline the study puts on trial: adapt the weights to the task,
then audit the trained model with the clinical SAE (src/stage2). It is
deliberately built to differ from the frozen probe in exactly one respect —
whether the weights move:

    same backbone, same chat template, same post-token pooling,
    same linear head shape, same training subsample.

so a gap between the arms cannot be blamed on a different backbone or a
different view of the text. Each arm is paired with the frozen probe that
shares its pooling: the mean-pooled arms against `linear_probe`, and the
attention-pooled arms (`AttentionPooledClassifier` below, which reuses every
other part of this file) against `attention_probe`.

The recipe is fixed a priori (configs/default.yaml), using the
stability-corrected settings from the few-sample fine-tuning literature
(bias-corrected AdamW, warmup, generous epochs) so that instability at small n
is a property of the method rather than of a handicapped baseline
(Mosbach et al. 2021; Zhang et al. 2021). Steps are capped and floored so tiny
budgets still converge and the largest stays affordable.
"""

import numpy as np
import torch
import torch.nn as nn

from src.utils.common import cuda_or_exit
from src.utils.data_prep import MODEL_REGISTRY, build_prompt


# The fine-tuning arms (+ anchor). `src.stage1.train` maps the readable method
# names onto these arm specs. Two things vary across the 2x2 and nothing else
# does: POOLING and HEAD INITIALIZATION.
#
#                     random head            probe-solved head
#   mean pooling      finetune_full_random   finetune_full_probe_init
#   attention pooling finetune_attn_random   finetune_attn_probe_init
#
# The attention column pools with a learned query. It is one subclass and one
# LP stage in this same file, reusing the training loop, batching, recipe and
# checkpoint format below -- so an attention arm differs from its mean-pooled
# partner in the pooling and in nothing else. Its LP stage is the
# frozen attention probe's own solver (src/probe/attention_probe.solve), which
# is what makes the fine-tuned and frozen attention arms comparable.
ARMS = {
    "finetune_full_random": dict(recipe="full", init="random", pooling="mean"),
    "finetune_full_probe_init": dict(recipe="full", init="probe", pooling="mean"),
    "finetune_attn_random": dict(recipe="full", init="random",
                                 pooling="attention"),
    "finetune_attn_probe_init": dict(recipe="full", init="probe",
                                     pooling="attention"),
    # The mid-layer-headed column. Same recipe, same pooling, same head
    # initializations -- the head is attached at the 0.65-depth working layer
    # instead of at the final one (`head_at="dense_probe_layer"`, resolved per
    # backbone in src/stage1/train.py because the index is model-dependent).
    #
    # WHY THE ARM EXISTS. The FT arms lose word AUPRC catastrophically
    # (results/dla_auprc.csv: full_ft 0.355 gemma / 0.214 llama, against the
    # frozen probes' 0.42-0.51) while winning or tying on detection AUROC. The
    # two facts have one candidate cause: the head sits on the FINAL layer,
    # where the residual stream has been mixed across positions for the whole
    # depth of the network, so a token's state is no longer about that token.
    # Aggregation done in-network buys the decision and costs the localisation.
    #
    # This column tests that by moving only the read-out site. The head reads
    # the same activation the frozen probes read -- cached index
    # DENSE_PROBE_LAYER, hidden_states[k], no final norm -- so a mid-layer arm
    # and its frozen partner sit on the same site and differ only in whether
    # the weights below it move. If the cause is depth, these arms keep FT's
    # AUROC and recover the probes' word AUPRC; if it is fine-tuning itself,
    # they lose word AUPRC anyway and the depth explanation is dead.
    #
    # `midlayer_attn_lp_ft` is the sharpest cell: its LP stage solves the
    # frozen attention probe AT THE PROBE'S OWN LAYER, so it does not merely
    # resemble the published `attention_probe` -- it starts there.
    # The random-head half of the mid-layer column. Removed on 2026-08-28 when
    # only the probe-init cells looked load-bearing, and restored on 2026-08-29:
    # without it there is no mid-layer arm for `midlayer_lp_ft` to be compared
    # against, so the head-initialization contrast could only be drawn at the
    # final layer. Its cells are the ones trained then, unchanged.
    "finetune_midlayer_random": dict(recipe="full", init="random",
                                     pooling="mean",
                                     head_at="dense_probe_layer"),
    "finetune_midlayer_attn_random": dict(recipe="full", init="random",
                                          pooling="attention",
                                          head_at="dense_probe_layer"),
    "finetune_midlayer_probe_init": dict(recipe="full", init="probe",
                                         pooling="mean",
                                         head_at="dense_probe_layer"),
    "finetune_midlayer_attn_probe_init": dict(recipe="full", init="probe",
                                              pooling="attention",
                                              head_at="dense_probe_layer"),
    # The LoRA column (2026-09-08). The same pooling x head-initialization
    # 2x2 with `recipe="lora"`: the backbone is frozen and low-rank adapters
    # (Hu et al. 2022) on every attention and MLP projection carry the update,
    # under the `lora` block of configs/default.yaml. The head (and the query)
    # train alongside the adapters at the LoRA learning rate. After training
    # the adapters are MERGED into the backbone (`merge_and_unload`), so the
    # checkpoint is a plain backbone plus head, byte-for-byte the format the
    # full arms write, and every Stage 2/3 consumer reads it unchanged.
    #
    # WHY THE ARM EXISTS. Full fine-tuning loses word AUPRC while keeping
    # AUROC; the mid-layer column showed the loss is not the read-out depth.
    # LoRA restricts the update to a low-rank subspace of each weight, which is
    # the standard practical alternative to moving every weight. If the
    # localisation loss is a property of moving the weights at all, LoRA loses
    # it too; if it is a property of the unconstrained update, LoRA keeps the
    # probe's localisation while matching FT's detection.
    "finetune_lora_random": dict(recipe="lora", init="random", pooling="mean"),
    "finetune_lora_probe_init": dict(recipe="lora", init="probe",
                                     pooling="mean"),
    "finetune_lora_attn_random": dict(recipe="lora", init="random",
                                      pooling="attention"),
    "finetune_lora_attn_probe_init": dict(recipe="lora", init="probe",
                                          pooling="attention"),
    # Span-supervised full fine-tuning (2026-09-08). `full_ft` / `attn_ft`
    # with `objective="span"`: the post BCE plus a token-level BCE on the
    # head's own projection w . h_t against the expert span (gold tokens
    # positive, every other post token negative; absent posts all negative;
    # present posts with no aligned span contribute only the post term). The
    # ranking Stage 2 scores IS w . h_t (times alpha), so this arm is told
    # directly where its evidence should sit. It is the upper reference for
    # plausibility: what fine-tuning localises when it is given the spans the
    # frozen probes never see. Random head only; the LP stage would solve a
    # head on post labels alone and then be asked to move it toward spans.
    # Last-token read-out (2026-09-09): full fine-tuning, random head, the
    # final post token's state as the post vector. The decoder-only default
    # probing/fine-tuning read-out, kept as the reference the mean-vs-
    # attention pooling contrast is read against. Random head only.
    "finetune_last_random": dict(recipe="full", init="random", pooling="last"),
    "finetune_span_random": dict(recipe="full", init="random", pooling="mean",
                                 objective="span"),
    "finetune_span_attn_random": dict(recipe="full", init="random",
                                      pooling="attention", objective="span"),
    # Not one of the 2x2 cells: a conventionally tuned full fine-tune, so the
    # matched-learning-rate design cannot be mistaken for a claim about how
    # standard fine-tuning behaves.
    "anchor": dict(recipe="full", init="random", pooling="mean",
                   use_anchor_lr=True),
}


# ------------------------------------------------------------------- batching


def build_inputs(tok, texts: list[str], max_len: int, spans=None):
    """Tokenize chat-templated posts and mark which tokens belong to the post.

    Mirrors src/utils/data_prep.encode_corpus exactly: the template is applied, and
    template tokens are excluded from pooling — they carry no post content and
    the probe never sees them either.

    `spans`, when given, is one (span_start, span_end) per post in POST
    character offsets, with a negative start meaning "no span", and a fourth
    tensor is returned: the gold token mask, 1 on every post token whose
    characters overlap the span -- the same overlap rule `src.probe.token_masks`
    scores plausibility with, shifted by the template prefix.
    """
    ids, post_masks, gold_masks = [], [], []
    for i, text in enumerate(texts):
        full, p0, p1 = build_prompt(tok, text)
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False,
                  truncation=True, max_length=max_len)
        mask = [1 if (a < p1 and b > p0 and b > a) else 0
                for a, b in enc["offset_mapping"]]
        if not any(mask):  # pathological truncation: fall back to all tokens
            mask = [1] * len(enc["input_ids"])
        ids.append(enc["input_ids"])
        post_masks.append(mask)
        if spans is not None:
            s, e = spans[i]
            if s is None or int(s) < 0:
                gold_masks.append([0] * len(mask))
            else:
                gs, ge = p0 + int(s), p0 + int(e)
                gold_masks.append([
                    1 if (m and a < ge and b > gs) else 0
                    for m, (a, b) in zip(mask, enc["offset_mapping"])])

    n = max(len(x) for x in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(ids), n), pad, dtype=torch.long)
    attn = torch.zeros((len(ids), n), dtype=torch.long)
    post = torch.zeros((len(ids), n), dtype=torch.float)
    gold = torch.zeros((len(ids), n), dtype=torch.float)
    for i, (seq, m) in enumerate(zip(ids, post_masks)):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        post[i, : len(m)] = torch.tensor(m, dtype=torch.float)
        if spans is not None:
            gold[i, : len(m)] = torch.tensor(gold_masks[i], dtype=torch.float)
    if spans is not None:
        return input_ids, attn, post, gold
    return input_ids, attn, post


# --------------------------------------------------------------------- model


class PooledClassifier(nn.Module):
    """Backbone -> mean-pool over post tokens -> linear head.

    Mean pooling is the probe's pooling. The head is then a linear probe over
    the same quantity the frozen mean-pool probe reads, so this arm and that
    probe differ only in whether the features underneath are allowed to move —
    and the post score is the token-count-weighted mean of its word scores, the
    identity the faithfulness argument rests on.
    """

    pooling = "mean"

    def __init__(self, backbone, hidden_size: int, head_layer: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        # None: the head reads the final layer -- the default and the only
        # thing this file did until 2026-08-28. An int k: the head reads
        # hidden_states[k], the SAME index convention the activation cache and
        # DENSE_PROBE_LAYER use (hidden_states[k] is the input to block k, i.e.
        # the output of block k-1), so a mid-layer arm reads exactly the site
        # its frozen partner reads.
        self.head_layer = head_layer
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def pooling_weights(self, h, post_mask):
        """alpha over post tokens -- [B, T], uniform here, zero off the mask.

        The pooling written as weights rather than as an average, so that
        anything decomposing the logit over tokens (`token_scores`,
        src/utils/dla.py) can ask the model how it pools instead of assuming.
        The attention arm overrides exactly this method.
        """
        m = post_mask.to(torch.float32)
        return m / m.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def token_states(self, input_ids=None, attention_mask=None, post_mask=None,
                     inputs_embeds=None):
        """Hidden states [B, T, H] at the head's site — the quantity the pooling
        reads, the features the LP stage is solved on, and the states
        `token_scores` decomposes the logit over. One method, so those three
        cannot end up reading different layers.

        `head_layer=None` returns `last_hidden_state`, which for these
        architectures is the final block's output AFTER the model's final norm.
        A mid-layer arm returns `hidden_states[k]` instead, which is NOT normed
        — matching the activation cache, which is written the same way
        (src/utils/data_prep.encode_corpus). The two are different quantities
        and the arm has to read the one its frozen partner reads.

        The LAST entry of `hidden_states` is `last_hidden_state`, normed, so
        indexing it would silently hand back the first quantity while the code
        claims the second. `truncate_backbone_at` keeps a block above the site
        precisely so that cannot happen, and the check below refuses the read
        rather than trusting that it did.
        """
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.head_layer is not None,
        )
        if self.head_layer is None:
            return out.last_hidden_state
        states = out.hidden_states
        if self.head_layer >= len(states) - 1:
            raise IndexError(
                f"head_layer {self.head_layer} but the backbone returned "
                f"{len(states)} hidden states, whose last entry "
                f"({len(states) - 1}) is post-final-norm and is not a raw "
                "block boundary; the site must be strictly below it")
        return states[self.head_layer]

    def pooled_from_states(self, h, post_mask):
        """Mean over post tokens. Template and padding positions are masked out,
        the same tokens data_prep.encode_corpus drops before writing the cache."""
        m = post_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def pooled(self, input_ids=None, attention_mask=None, post_mask=None,
               inputs_embeds=None):
        h = self.token_states(input_ids, attention_mask, post_mask, inputs_embeds)
        return self.pooled_from_states(h, post_mask)

    def logits_and_states(self, input_ids=None, attention_mask=None,
                          post_mask=None, inputs_embeds=None):
        """(logit [B], token states [B, T, H]) from ONE forward pass.

        The span-supervised objective needs both the decision and the states
        its token term is written on; reading them from a single pass keeps
        the two terms on the same computation rather than on two runs of a
        checkpointed backbone.
        """
        h = self.token_states(input_ids, attention_mask, post_mask, inputs_embeds)
        pooled = self.pooled_from_states(h, post_mask)
        return self.head(pooled.float()).squeeze(-1), h

    def forward(self, input_ids=None, attention_mask=None, post_mask=None,
                inputs_embeds=None):
        pooled = self.pooled(input_ids, attention_mask, post_mask, inputs_embeds)
        return self.head(pooled.float()).squeeze(-1)


class AttentionPooledClassifier(PooledClassifier):
    """Backbone -> softmax pooling under a learned query -> linear head.

    The mean-pooled parent's `forward`, `token_states` and head are reused
    unchanged; the ONLY override is how the post tokens are weighted. That is
    the whole of the attention column of the 2x2, and it is why `attn_ft` can
    be read as `full_ft` with one thing changed.

    Scores, the softmax and the weighted sum are all taken in float32 even
    though the backbone runs in bf16: alpha is a distribution over up to 512
    positions, and bf16's 8 mantissa bits round small weights hard enough to
    move the pooled vector. The frozen probe's solver is float32 too, which is
    what lets an LP solution transfer into this class rather than merely land
    near it.

    WHAT DECOMPOSITION SURVIVES
    ---------------------------
    The same one attention pooling leaves the frozen probe with, and no more.
    With alpha_t = softmax_t(h_t . q / sqrt(H)) over the post tokens,

        logit = sum_t alpha_t (w . h_t) + b

    so every token still has an additive share of the decision and the shares
    still sum to it -- but alpha is a softmax over the whole post, so a token's
    share depends on what else the post says. It is a contribution IN CONTEXT,
    not an intrinsic word score, exactly as in src/probe/attention_probe.py.
    `token_scores` reads the weights off the model rather than assuming uniform
    ones, so it serves both poolings, and `verify_decomposition` enforces the
    identity instead of asserting it.
    """

    pooling = "attention"

    def __init__(self, backbone, hidden_size: int, init_scale: float = 0.01,
                 head_layer: int | None = None):
        super().__init__(backbone, hidden_size, head_layer)
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=init_scale)

    def pooling_weights(self, h, post_mask):
        """alpha over post tokens -- [B, T] float32, zero off the post mask."""
        return self._pool(h, post_mask)[1]

    def _pool(self, h, post_mask):
        """(pooled [B, H], alpha [B, T]), both float32, from token states."""
        hf = h.float()
        scores = (hf @ self.query) / (self.hidden_size ** 0.5)
        alpha = torch.softmax(
            scores.masked_fill(post_mask <= 0, float("-inf")), dim=1)
        return (hf * alpha.unsqueeze(-1)).sum(1), alpha

    def pooled_from_states(self, h, post_mask):
        return self._pool(h, post_mask)[0]


class LastTokenClassifier(PooledClassifier):
    """Backbone -> the LAST post token's state -> linear head.

    The decoder-only default read-out (the position that has attended to the
    whole post), added 2026-09-09 as the reference the pooling contrast was
    missing. alpha is a one-hot on the final post token, so the additive
    decomposition still holds trivially -- and says nothing: every other
    token's share is zero. This arm therefore has NO own word ranking; its
    Stage 2 cells come from gradient x input through the backbone
    (src.stage2.gradient_input), which is where the credit the last position
    absorbed from earlier words is handed back to them.
    """

    pooling = "last"

    def pooling_weights(self, h, post_mask):
        m = post_mask.to(torch.float32)
        # Index of the last on-mask position per row: cumulative count equals
        # the row total exactly once, at that position.
        last = (m.cumsum(1) == m.sum(1, keepdim=True)) & (m > 0)
        return last.to(torch.float32)

    def pooled_from_states(self, h, post_mask):
        alpha = self.pooling_weights(h, post_mask).to(h.dtype)
        return (h * alpha.unsqueeze(-1)).sum(1)


def attn_implementation_for(model_id_or_config) -> str:
    """"eager" when the architecture soft-caps attention logits, else "sdpa".

    Gemma-2 caps attention logits at 50.0, but `sdpa_attention_forward` takes
    `softcap` only through `**kwargs` and never applies it — the cap is dropped
    silently, with no warning. That is not a speed/accuracy trade: it makes the
    forward pass a *different function*. It matters here because the Gemma Scope
    SAEs this study audits with were trained on the true, soft-capped residual
    stream, so reading an uncapped stream would feed the SAE activations from a
    model it never saw, and the mismatch would be invisible in every metric.

    Architectures without soft-capping (Llama) are unaffected and keep
    SDPA's memory advantage.
    """
    import os

    from transformers import AutoConfig

    config = (
        AutoConfig.from_pretrained(model_id_or_config)
        if isinstance(model_id_or_config, (str, os.PathLike))
        else model_id_or_config
    )
    softcap = getattr(config, "attn_logit_softcapping", None)
    return "eager" if softcap else "sdpa"


def build_classifier(backbone, hidden: int, pooling: str, cfg: dict,
                     head_layer: int | None = None):
    """Wrap a backbone in the classifier this arm's pooling calls for."""
    if pooling == "mean":
        return PooledClassifier(backbone, hidden, head_layer)
    if pooling == "attention":
        init_scale = float(cfg.get("attention_pool", {}).get("init_scale", 0.01))
        return AttentionPooledClassifier(backbone, hidden, init_scale,
                                         head_layer)
    if pooling == "last":
        return LastTokenClassifier(backbone, hidden, head_layer)
    raise ValueError(f"unknown pooling {pooling!r}; use 'mean', 'attention' "
                     "or 'last'")


def truncate_backbone_at(base, head_layer: int):
    """Keep `head_layer + 1` blocks and drop the rest. Exact, not approximate.

    A decoder is a feedforward stack, so blocks above k cannot influence
    hidden_states[k]; deleting them leaves that tensor identical and removes a
    forward pass that produced nothing the loss reads. On a mid-layer arm that
    is ~13 of gemma-2-2b's 26 blocks and ~8 of llama31-8b's 32, paid on every
    step of every fold, so it is worth doing rather than documenting.

    THE +1 IS LOAD-BEARING, AND THIS IS THE SECOND OFF-BY-ONE IN THE REPO
    ---------------------------------------------------------------------
    `output_hidden_states` does NOT return N+1 raw block boundaries. The LAST
    entry is `last_hidden_state`, i.e. the final block's output with the
    model's final norm ALREADY APPLIED; every earlier entry is raw. So on a
    stack cut to exactly `head_layer` blocks, `hidden_states[head_layer]` is
    the last entry and comes back normed -- a different tensor from the one the
    full stack returns at that index, and a different one from what the
    activation cache holds. Measured on transformers 5.15: max abs difference
    3.34 on a random 8-block Llama, with indices 0..4 bit-identical and only
    index 5 wrong. Nothing raises; the shapes match and the arm simply trains
    on a quantity its frozen partner never saw.

    Keeping one extra block puts `head_layer` strictly inside the sequence, so
    it is always a raw boundary. The extra block runs in the forward pass and
    is frozen: nothing reads its output, so it takes no gradient anyway, and
    freezing says so rather than leaving a reader to work it out. The invariant
    it buys -- head_layer < len(hidden_states) - 1 -- is what
    `PooledClassifier.token_states` enforces on every call.

    `layer_idx` travels with each block, so Gemma-2's alternating
    sliding-window / full attention pattern is unchanged by truncation.
    """
    layers = getattr(base, "layers", None)
    if layers is None:      # an architecture that names its stack differently
        raise AttributeError(
            f"{type(base).__name__} has no `.layers`; a mid-layer arm on this "
            "backbone needs its block list named here before truncating")
    keep = head_layer + 1
    if keep > len(layers):
        raise ValueError(
            f"head_layer {head_layer} needs {keep} of the backbone's "
            f"{len(layers)} blocks: the site must stay strictly below the "
            "final one, whose hidden state comes back normed")
    base.layers = layers[:keep]
    base.config.num_hidden_layers = keep
    # `layer_types` is a PER-BLOCK list (Gemma-2 alternates sliding-window and
    # full attention down the stack), and transformers validates it against
    # `num_hidden_layers` when the config is saved -- so a truncated stack whose
    # list was left at full length trains for ten minutes and then dies in
    # `save_checkpoint`, after the work and before the weights. Slicing the
    # front of the list keeps every retained block's own type, which is what
    # makes the reloaded checkpoint the same function as the trained one.
    types = getattr(base.config, "layer_types", None)
    if types is not None:
        base.config.layer_types = list(types)[:keep]
    for prm in base.layers[head_layer].parameters():
        prm.requires_grad_(False)   # runs, is never read, must never train
    return base


def load_backbone(model_name: str, cfg: dict, seed: int, recipe: str = "full",
                  pooling: str = "mean", head_layer: int | None = None):
    """The backbone with every parameter trainable — the recipe Kumar et al.
    2022's feature-distortion argument is actually stated for, and therefore
    the one the LP-FT claim needs.

    `recipe="lora"` freezes every backbone weight and attaches low-rank
    adapters (`cfg["lora"]`) to the projections it names; only the adapters,
    the head and (for an attention arm) the query take gradients. The
    adapters start at zero, so at initialization the model IS the pretrained
    backbone and an LP stage solved on it is exact, as it is for the full arm.
    """
    from transformers import AutoModel, AutoTokenizer

    assert recipe in ("full", "lora"), f"unknown recipe {recipe!r}"
    torch.manual_seed(seed)
    model_id = MODEL_REGISTRY[model_name]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # SDPA avoids materializing the [batch, heads, T, T] score matrix; the layer
    # activations still do not fit an 8B model at batch 16, so checkpointing
    # stays on (configs/default.yaml: full_finetune.grad_checkpointing). Measured:
    # ~0.8 s/step with both, versus OOM with either one alone.
    base = AutoModel.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation=attn_implementation_for(model_id))
    base.config.use_cache = False
    if cfg.get("grad_checkpointing"):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    hidden = base.config.hidden_size  # read before any wrapper hides the config
    if head_layer is not None:
        truncate_backbone_at(base, head_layer)
    if recipe == "lora":
        from peft import LoraConfig, get_peft_model

        if "lora" not in cfg:
            raise KeyError("recipe='lora' needs the `lora` block of "
                           "configs/default.yaml in the recipe cfg")
        lcfg = cfg["lora"]
        for p in base.parameters():
            p.requires_grad_(False)
        base = get_peft_model(base, LoraConfig(
            r=int(lcfg["r"]), lora_alpha=float(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]),
            target_modules=list(lcfg["target_modules"]), bias="none"))
        model = build_classifier(base, hidden, pooling, cfg, head_layer)
        return cuda_or_exit(model, what=f"the {model_name} backbone"), tok
    for p in base.parameters():
        p.requires_grad_(True)
    if head_layer is not None:
        # ORDER MATTERS. `truncate_backbone_at` freezes the one block it keeps
        # ABOVE the head site, and the blanket `requires_grad_(True)` above
        # would silently undo that -- the freeze is written where the reason
        # for it is, and re-applied here where it survives. Nothing reads that
        # block's output, so it takes no gradient either way and no published
        # number moves; what the re-freeze buys is that the invariant the arm
        # is documented under is the invariant it actually runs under, and that
        # the checkpointed forward does not carry a block whose recomputation
        # is pure waste.
        for prm in base.layers[head_layer].parameters():
            prm.requires_grad_(False)
    model = build_classifier(base, hidden, pooling, cfg, head_layer)
    return cuda_or_exit(model, what=f"the {model_name} backbone"), tok


# ------------------------------------------------- head initialization (LP-FT)


# --------------------------------------------------- the LP stage of LP-FT
#
# Kumar et al. 2022: fine-tuning through a randomly initialized head sends
# large, uninformative gradients into the feature extractor and warps the
# pre-trained geometry that carries out-of-distribution generalization. Solving
# the head first, on frozen features, means the FT phase starts near-optimal and
# only has to move the backbone a little. The initialization below is an exact
# solve on the frozen representation, not a warm start.
#
# The features are read at the FINAL layer, deliberately: that is the quantity
# the head actually sits on. Fitting the init at the pinned 0.65-depth layer the
# frozen evaluation probes use would build a layer mismatch into the arm;
# the pin governs the evaluation probes, not this stage.


@torch.no_grad()
def token_scores(model, input_ids, attention_mask, post_mask) -> "np.ndarray":
    """Each post token's exact additive share of this arm's own logit.

    The head is linear on the pooled representation, and pooling is a weighted
    sum of the post tokens, so

        logit = sum_t alpha_t (w . h_t) + b

    holds exactly and alpha_t (w . h_t) is token t's own contribution. Under
    mean pooling alpha_t = 1/T and the contribution is a property of the token
    alone; under the attention arm's learned query alpha is a softmax over the
    whole post, so the identity still holds but the contribution is one IN
    CONTEXT (`AttentionPooledClassifier`). The weights come from the
    model, so this stays one decomposition rather than two.

    This is the same decomposition src.stage2.dla_auprc scores word AUPRC over,
    so plausibility and the word-deletion faithfulness arm rank by one ranking.
    Returns [T] over the batch's single sequence, zero off the post mask.
    """
    model.eval()
    h = model.token_states(input_ids, attention_mask, post_mask)
    w = model.head.weight[0].to(h.dtype)
    projected = (h @ w).float()                      # [B, T]
    alpha = model.pooling_weights(h, post_mask).to(projected.dtype)
    return (projected * alpha)[0].detach().cpu().numpy()


@torch.no_grad()
def pooled_features(model, tok, texts: list[str], cfg: dict, batch_size: int = 8):
    """Frozen pooled representations — the exact quantity the head reads."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        ids, attn, post = build_inputs(tok, texts[i : i + batch_size], cfg["max_len"])
        p = model.pooled(ids.cuda(), attn.cuda(), post.cuda())
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def init_head_from_probe(model, tok, texts, labels, cfg, seed) -> dict:
    """LP stage: solve this arm's readout on the frozen representation.

    Mean pooling has a closed form on the pooled vectors and is solved below.
    Attention pooling has no pooled vector to solve on — the pooling is part of
    what is being fitted — so that arm runs the frozen attention probe's own
    solver instead. Returns the stage's provenance for the recipe sidecar.
    """
    if getattr(model, "pooling", "mean") == "attention":
        return _init_attention_head_from_probe(
            model, tok, texts, labels, cfg, seed)
    return _init_mean_head_from_probe(model, tok, texts, labels, cfg, seed)


def _init_mean_head_from_probe(model, tok, texts, labels, cfg, seed) -> dict:
    """Mean pooling: solve the linear head exactly on frozen pooled features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = pooled_features(model, tok, texts, cfg)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(X), labels)
    # Fold the standardizer into the linear layer so the head stays a plain
    # affine map of the pooled vector.
    scale = np.maximum(scaler.scale_, 1e-8)
    w = clf.coef_[0] / scale
    b = float(clf.intercept_[0] - (clf.coef_[0] * scaler.mean_ / scale).sum())
    with torch.no_grad():
        model.head.weight.copy_(
            torch.tensor(w, dtype=model.head.weight.dtype).unsqueeze(0)
        )
        model.head.bias.fill_(b)
    pred = ((X @ w + b) > 0).astype(int)
    return {"lp_train_acc": float((pred == np.asarray(labels)).mean()),
            "lp_solver": "sklearn logistic on frozen mean-pooled features"}


# ------------------------------------- the LP stage of AP-LP-FT (attn_lp_ft)
#
# `_init_attention_head_from_probe` calls src.probe.attention_probe.solve, the
# exact objective the frozen attention probe is fitted with, on final-layer
# states pulled out of the live backbone. Two consequences worth keeping:
#
#   * the LP stage's starting point IS the frozen attention probe (at the final
#     layer), so "what does fine-tuning move?" is asked against a known point
#     rather than against a re-derived approximation of one;
#   * a change to the frozen probe's solver changes this arm too, by
#     construction. There is one solver in the repo, not a probe-side copy and
#     a fine-tune-side copy that drift.
#
# Features are read at the FINAL layer, matching the mean-pooled LP stage above
# and for the same reason: that is the quantity the head actually sits on. The
# 0.65-depth pin governs the frozen EVALUATION probes, not this stage.
#
# The query is trained at the FT phase's single learning rate, alongside the
# backbone and the head -- not at the frozen probe's `lp_lr`, which belongs to
# the LP stage. Splitting parameter groups would add a second difference to an
# arm that exists to isolate one.


@torch.no_grad()
def token_features(model, tok, texts: list[str], cfg: dict,
                   batch_size: int = 8) -> list[np.ndarray]:
    """One [T_i, H] float16 array per post: its POST tokens' frozen states.

    The attention-pooled LP stage cannot work from pooled vectors the way
    `pooled_features` does -- the pooling is what is being solved -- so the
    tokens have to survive the pass. They are kept as float16 on the host and
    upcast per minibatch inside the solver: at ~200 post tokens a post, the
    full training pool is ~3 GB on gemma-2-2b (2304 dims) and ~5 GB on
    llama31-8b (4096), against twice that in float32.
    """
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        ids, attn, post = build_inputs(tok, texts[i: i + batch_size],
                                       cfg["max_len"])
        ids, attn, post = ids.cuda(), attn.cuda(), post.cuda()
        h = model.token_states(ids, attn, post).float().cpu().numpy()
        keep = post.cpu().numpy() > 0
        for row in range(h.shape[0]):
            out.append(h[row][keep[row]].astype(np.float16))
    return out


def _init_attention_head_from_probe(model, tok, texts, labels, cfg: dict,
                                    seed: int) -> dict:
    """Attention pooling: solve (q, w, b) on the frozen token states.

    `cfg` is the recipe block, which for an attention arm carries the frozen
    probe's solver settings under `attention_pool` (src/stage1/train.py copies
    them in, so they land in the run's recipe sidecar instead of being read
    from a global at some unrecorded moment).

    Returns the LP stage's provenance: its own training accuracy, and the
    largest disagreement between the installed head and the probe it came from.
    """
    from src.probe import attention_probe

    if "attention_pool" not in cfg:
        raise KeyError(
            "the attention-pooled LP stage needs the frozen probe's solver "
            "settings; src.stage1.train copies cfg['attention_pool'] into the "
            "recipe block for attention arms")
    y = np.asarray(labels).astype(int)
    feats = token_features(model, tok, texts, cfg)
    probe = attention_probe.solve(feats, y, seed,
                                  {"attention_pool": cfg["attention_pool"]})
    with torch.no_grad():
        model.query.copy_(torch.tensor(probe["q"], dtype=model.query.dtype))
        model.head.weight.copy_(
            torch.tensor(probe["w_raw"],
                         dtype=model.head.weight.dtype).unsqueeze(0))
        model.head.bias.fill_(float(probe["b"]))
    return {
        "lp_train_acc": float(probe["train_accuracy"]),
        "lp_solver": probe["solver"],
        # Enforced: did (q, w, b) actually land in the module?
        "lp_transfer_max_abs": verify_head_transfer(model, feats, probe),
        # Measured, not enforced: what the backbone re-run costs on top.
        "lp_live_forward_max_abs": live_forward_drift(model, tok, texts, cfg,
                                                      feats, probe),
    }


# The LP stage's transfer is checked in two pieces, because two different
# things can go wrong and only one of them is this module's fault.
#
#   verify_head_transfer  ENFORCED, tight. Runs the module's OWN pooling and
#       head over the SAME cached features the probe was solved on. Nothing but
#       float32-vs-float64 separates the two sides, so this is exact to ~1e-4 —
#       and it is what catches a genuinely botched install: a transposed w, a
#       query left at its random init, a dropped bias, a wrong 1/sqrt(H).
#
#   live_forward_drift    MEASURED, reported, warned. Re-runs the backbone and
#       compares against the same probe score. This one CANNOT be tight, and an
#       earlier 5e-2 assertion on it failed at 0.0709 on the first real cell:
#         * `token_features` encodes at batch 8 with padding, a live check at
#           batch 1 with none, and padded and unpadded attention do not produce
#           bit-identical states;
#         * the backbone is bf16 (8 mantissa bits), and alpha is a softmax of a
#           2304-dim dot product, so a per-element rounding difference of ~0.4%
#           is amplified through exp() before it reaches the pooled vector.
#       Both are properties of the forward pass, not of the head that was just
#       installed. Failing the run on them told us nothing about the transfer;
#       the number is worth keeping in the sidecar, so it is kept there.


@torch.no_grad()
def verify_head_transfer(model, feats, probe, n: int = 64,
                         atol: float = 1e-4) -> float:
    """Largest |module score - probe score| over `n` posts, same features both sides.

    The installed head is supposed to BE the solved probe. Here it is asked to
    prove that on the probe's own inputs: the cached token states go through
    `AttentionPooledClassifier._pool` and `head` instead of through
    `attention_probe.post_score`, with no backbone in the loop. What is left
    between them is the module's float32 against the probe's float64, so the
    tolerance can be tight enough to mean something.
    """
    from src.probe import attention_probe

    model.eval()
    device = next(model.parameters()).device
    worst = 0.0
    for i in range(min(n, len(feats))):
        vecs = np.asarray(feats[i], dtype=np.float32)
        if not len(vecs):
            continue
        h = torch.from_numpy(vecs).unsqueeze(0).to(device)
        post = torch.ones(1, h.shape[1], dtype=torch.float32, device=device)
        pooled, _ = model._pool(h, post)
        got = float(model.head(pooled.float()).squeeze(-1).item())
        want = attention_probe.post_score(
            probe, {"vecs": vecs.astype(np.float64)})
        worst = max(worst, abs(got - want))
    if worst > atol:
        raise AssertionError(
            f"the installed attention head disagrees with the probe it was "
            f"solved from by {worst:.3g} log-odds on the probe's own cached "
            f"features (tolerance {atol}). No backbone is involved in this "
            f"comparison, so this is an installation error — check q, w, b "
            f"and the 1/sqrt(H) scaling, not the forward pass")
    return float(worst)


@torch.no_grad()
def live_forward_drift(model, tok, texts, cfg: dict, feats, probe,
                       n: int = 16, warn_at: float = 0.5) -> float:
    """Largest |live logit - probe score| over `n` posts. Reported, never fatal.

    What the backbone re-run adds on top of `verify_head_transfer`: bf16
    rounding amplified through the pooling softmax, plus batched-with-padding
    encoding against this unbatched one. Recorded on every attention LP stage
    so the magnitude is a measurement in the sidecar rather than a number
    someone has to guess at when setting a tolerance.
    """
    from src.probe import attention_probe

    model.eval()
    worst = 0.0
    for i in range(min(n, len(texts))):
        ids, attn, post = build_inputs(tok, [texts[i]], cfg["max_len"])
        got = float(model(ids.cuda(), attn.cuda(), post.cuda()).float().item())
        want = attention_probe.post_score(
            probe, {"vecs": np.asarray(feats[i], dtype=np.float64)})
        worst = max(worst, abs(got - want))
    if worst > warn_at:
        print(f"    WARNING the live forward pass differs from the solved "
              f"probe by {worst:.3g} log-odds. verify_head_transfer passed, so "
              f"the head installed correctly; this is the backbone re-run. "
              f"Large values point at nondeterminism (dropout left on, "
              f"use_cache) rather than at the LP stage.", flush=True)
    return float(worst)


# ------------------------------------------------------------------ training


def n_steps(n_examples: int, cfg: dict) -> int:
    per_epoch = max(1, int(np.ceil(n_examples / cfg["batch_size"])))
    return int(np.clip(per_epoch * cfg["epochs"], cfg["min_steps"], cfg["max_steps"]))


def span_term(model, h, post_mask, gold, has_target, balanced: bool):
    """The token-level term of the span-supervised objective, one batch.

    BCE on the head's own projection w . h_t -- the quantity Stage 2 ranks
    words by (times the pooling weight) -- against the gold token mask. With
    `balanced` the gold and non-gold tokens of a post are averaged separately
    and then together, so the ~8% of tokens inside a span are not drowned by
    the rest; a post with no gold tokens (an absent post) contributes its
    non-gold mean alone. Posts without a target (`has_target` 0: present with
    no aligned span) contribute nothing. Returns the mean over targeted posts.
    """
    w = model.head.weight[0].to(h.dtype)
    proj = (h @ w).float()                                   # [B, T]
    per = nn.functional.binary_cross_entropy_with_logits(
        proj, gold.float(), reduction="none")
    on = post_mask > 0
    pos, neg = on & (gold > 0), on & (gold <= 0)
    if balanced:
        pos_mean = (per * pos).sum(1) / pos.sum(1).clamp(min=1)
        neg_mean = (per * neg).sum(1) / neg.sum(1).clamp(min=1)
        per_post = torch.where(pos.any(1), 0.5 * (pos_mean + neg_mean),
                               neg_mean)
    else:
        per_post = (per * on).sum(1) / on.sum(1).clamp(min=1)
    keep = has_target.float()
    return (per_post * keep).sum() / keep.sum().clamp(min=1)


def train(model_name: str, texts: list[str], labels: np.ndarray, cfg: dict,
          seed: int, init: str = "random", recipe: str = "full",
          pooling: str = "mean", lr: float | None = None, log_every: int = 0,
          head_layer: int | None = None, objective: str = "bce",
          spans=None):
    """Fine-tune one arm. Returns (model, tokenizer, info).

    `objective="span"` adds `span_term` to the post BCE, weighted by
    cfg["span_supervision"]["weight"]; `spans` then carries one
    (span_start, span_end) per training post (negative start = none). A
    recipe of "lora" trains adapters, the head and the query at
    cfg["lora"]["lr"] with plain AdamW, and merges the adapters into the
    backbone before returning, so the returned model has the same module
    structure as a full arm and is saved and reloaded the same way.

    `cfg` is the recipe block cfg["full_finetune"]. `init` and `pooling` are
    the two factors of the 2x2 and the only things that vary across its four
    cells; `pooling="attention"` adds a learned query, which is trained at the
    same single learning rate as the backbone and the head so that the pooling
    is the only difference. `lr` overrides the block's learning rate, which is
    how the conventional-recipe anchor cell is run without disturbing the
    matched-LR comparison.

    `head_layer` is the third factor, added 2026-08-28: None attaches the head
    at the final layer (every arm published so far), an int attaches it at that
    cached layer index and truncates the blocks above it. It changes where the
    readout sits and nothing else -- same recipe, same LR, same step budget --
    so a mid-layer arm is its final-layer partner with one thing moved.
    """
    from transformers import get_linear_schedule_with_warmup

    assert init in ("random", "probe"), f"unknown init {init!r}"
    assert pooling in ("mean", "attention", "last"), f"unknown pooling {pooling!r}"
    assert objective in ("bce", "span"), f"unknown objective {objective!r}"
    if objective == "span":
        if spans is None or len(spans) != len(texts):
            raise ValueError("objective='span' needs one (start, end) per "
                             "training post")
        if "span_supervision" not in cfg:
            raise KeyError("objective='span' needs the `span_supervision` "
                           "block of configs/default.yaml in the recipe cfg")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, tok = load_backbone(model_name, cfg, seed, recipe=recipe,
                               pooling=pooling, head_layer=head_layer)

    info = {"init": init, "recipe": recipe, "pooling": pooling,
            "head_layer": head_layer, "objective": objective}
    if recipe == "lora":
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        info["lora"] = dict(cfg["lora"])
        info["trainable_parameters"] = int(trainable)
        print(f"    lora: r={cfg['lora']['r']} on "
              f"{list(cfg['lora']['target_modules'])}, "
              f"{trainable / 1e6:.2f}M trainable parameters", flush=True)
    if objective == "span":
        has = np.array([s is not None and int(s[0]) >= 0 for s in spans])
        info["span_supervision"] = dict(cfg["span_supervision"])
        info["n_train_with_span"] = int(has.sum())
        info["n_train_present"] = int(np.asarray(labels).astype(int).sum())
    if init == "probe":
        info.update(init_head_from_probe(model, tok, texts, labels, cfg, seed))
    if pooling == "attention":
        # Where the pooling starts, so the after-FT reading below has something
        # to be a change FROM.
        info["alpha_at_init"] = attention_entropy(model, tok, texts, cfg)
        print(f"    alpha at init: {_alpha_note(info['alpha_at_init'])}",
              flush=True)

    steps = n_steps(len(texts), cfg)
    if lr is None:
        lr = cfg["lora"]["lr"] if recipe == "lora" else cfg["lr"]
    info["lr"] = float(lr)
    params = [p for p in model.parameters() if p.requires_grad]
    if recipe == "lora":
        # The 8-bit optimizer exists to fit an 8B full fine-tune on one card;
        # the adapters are a few tens of MB, so the standard optimizer of the
        # LoRA recipe is used and recorded.
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        info["optimizer"] = "adamw"
    elif cfg.get("optimizer") == "adamw8bit":
        # 8-bit Adam states are what make full fine-tuning of an 8B backbone fit
        # a single 80GB card: bf16 weights 16.4 + bf16 grads 16.4 + 8-bit
        # moments ~16.4 = ~49 GB, against ~131 GB for fp32-master AdamW.
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=cfg["weight_decay"])
        info["optimizer"] = "adamw8bit"
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        info["optimizer"] = "adamw"
    sched = get_linear_schedule_with_warmup(
        opt, int(cfg["warmup_ratio"] * steps), steps
    )
    lossf = nn.BCEWithLogitsLoss()
    y = torch.tensor(labels, dtype=torch.float)
    rng = np.random.default_rng(seed)
    order: list[int] = []
    if objective == "span":
        span_weight = float(cfg["span_supervision"]["weight"])
        balanced = bool(cfg["span_supervision"].get("balanced", True))
        has_target = torch.tensor(
            [s is not None and int(s[0]) >= 0 for s in spans],
            dtype=torch.float)

    model.train()
    for step in range(steps):
        if len(order) < cfg["batch_size"]:
            order += rng.permutation(len(texts)).tolist()
        idx, order = order[: cfg["batch_size"]], order[cfg["batch_size"] :]
        batch_texts = [texts[i] for i in idx]
        if objective == "span":
            ids, attn, post, gold = build_inputs(
                tok, batch_texts, cfg["max_len"], spans=[spans[i] for i in idx])
            ids, attn, post, gold = ids.cuda(), attn.cuda(), post.cuda(), gold.cuda()
            logits, h = model.logits_and_states(ids, attn, post)
            post_loss = lossf(logits, y[idx].cuda())
            tok_loss = span_term(model, h, post, gold, has_target[idx].cuda(),
                                 balanced)
            loss = post_loss + span_weight * tok_loss
        else:
            ids, attn, post = build_inputs(tok, batch_texts, cfg["max_len"])
            logits = model(ids.cuda(), attn.cuda(), post.cuda())
            loss = lossf(logits, y[idx].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if log_every and step % log_every == 0:
            extra = (f" (post {post_loss.item():.4f}, span {tok_loss.item():.4f})"
                     if objective == "span" else "")
            print(f"    step {step}/{steps} loss {loss.item():.4f}{extra}",
                  flush=True)

    info["steps"] = steps
    info["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    model.eval()
    if recipe == "lora":
        # Fold the adapters into the frozen weights. From here on the model is
        # a plain backbone plus readout: `save_checkpoint` writes it, and
        # `load_checkpoint` rebuilds it, exactly as for a full arm. Measured
        # and recorded: the largest logit change the merge itself introduces
        # (bf16 rounding of W + BA against W and BA applied separately).
        # The change is read against the spread of the logits it moves, since
        # a probe-initialised head puts the logits on a scale a random one
        # does not. bf16 cannot hold W + BA exactly: entries of BA below half
        # an ulp of W vanish and the rest are quantised, so the merged model
        # is the trained function up to this recorded amount. Every score,
        # threshold and Stage 2/3 number is computed AFTER the merge, on the
        # model the checkpoint actually holds.
        sample = list(texts)[:64]
        before = predict(model, tok, sample, cfg)
        model.backbone = model.backbone.merge_and_unload()
        after = predict(model, tok, sample, cfg)
        change = np.abs(before - after)
        info["lora_merge_max_abs"] = float(change.max())
        info["lora_merge_mean_abs"] = float(change.mean())
        info["lora_merge_logit_std"] = float(np.std(before))
        print(f"    lora: adapters merged; logit change max "
              f"{info['lora_merge_max_abs']:.3g}, mean "
              f"{info['lora_merge_mean_abs']:.3g}, against a logit std of "
              f"{info['lora_merge_logit_std']:.3g}", flush=True)
    if pooling == "attention":
        # The arm's own falsification test: if the query has relaxed toward
        # uniform, the backbone absorbed the pooling and this arm is full_ft
        # wearing a softmax (`attention_entropy`).
        info["alpha_after_ft"] = attention_entropy(model, tok, texts, cfg)
        print(f"    alpha after FT: {_alpha_note(info['alpha_after_ft'])}",
              flush=True)
    return model, tok, info


def _alpha_note(alpha: dict) -> str:
    """One line of the pooling diagnostic, for the training log."""
    if not alpha:
        return "not measured"
    return (f"normalized entropy {alpha['normalized_entropy']:.3f} "
            f"(1.0 = uniform), spread over {alpha['effective_tokens']:.1f} "
            f"tokens, largest weight {alpha['max_weight']:.3f}")


# ------------------------------------------- did the pooling do anything?


# How many training posts the alpha diagnostic is measured over. Bounded because
# it runs twice per cell and is a diagnostic, not a metric.
ENTROPY_SAMPLE = 64


@torch.no_grad()
def attention_entropy(model, tok, texts: list[str], cfg: dict,
                      n: int = ENTROPY_SAMPLE, batch_size: int = 8) -> dict:
    """How concentrated the learned pooling actually is, in three numbers.

    The attention arms' own falsification test. A fine-tuned backbone can
    emulate mean pooling by writing the decision direction into every post
    token, in which case the query has nothing left to do and alpha relaxes
    toward uniform -- and the honest conclusion is that attention pooling buys
    nothing once the weights move. Measured at LP init and again after the FT
    phase, so the sidecar says which happened instead of leaving it to be
    inferred from an AUROC that would look the same either way.

        normalized_entropy   H(alpha) / log(T), 1.0 exactly at uniform
        effective_tokens     exp(H(alpha)): how many tokens alpha is spread over
        max_weight           the single largest alpha in the post

    Means over `n` posts, taken in the order given (deterministic).
    """
    model.eval()
    texts = list(texts)[:n]
    rows = []
    for i in range(0, len(texts), batch_size):
        ids, attn, post = build_inputs(tok, texts[i: i + batch_size],
                                       cfg["max_len"])
        ids, attn, post = ids.cuda(), attn.cuda(), post.cuda()
        h = model.token_states(ids, attn, post)
        alpha = model.pooling_weights(h, post).double().cpu().numpy()
        keep = (post > 0).cpu().numpy()
        for row in range(alpha.shape[0]):
            a = alpha[row][keep[row]]
            a = a[a > 0]
            if len(a) < 2:
                continue
            entropy = float(-(a * np.log(a)).sum())
            rows.append((entropy / np.log(len(a)), np.exp(entropy), a.max()))
    if not rows:
        return {}
    normalized, effective, largest = (np.array(c) for c in zip(*rows))
    return {"normalized_entropy": float(normalized.mean()),
            "effective_tokens": float(effective.mean()),
            "max_weight": float(largest.mean()),
            "n_posts": len(rows)}


@torch.no_grad()
def verify_decomposition(model, tok, texts: list[str], cfg: dict,
                         atol: float = 1e-3) -> float:
    """Largest |sum_t alpha_t (w . h_t) + b - logit| over `texts`.

    Holds for either pooling: `token_scores` reads alpha off the model, so this
    checks the mean-pooled identity and the weakened attention-pooled one with
    the same code. Not called by the training path -- it is the check to reach
    for when a decomposition is in doubt.
    """
    model.eval()
    worst = 0.0
    for text in texts:
        ids, attn, post = build_inputs(tok, [text], cfg["max_len"])
        ids, attn, post = ids.cuda(), attn.cuda(), post.cuda()
        terms = token_scores(model, ids, attn, post)
        logit = float(model(ids, attn, post).float().item())
        worst = max(worst,
                    abs(float(terms.sum()) + float(model.head.bias) - logit))
    if worst > atol:
        raise AssertionError(
            f"pooled decomposition broken: contributions and logit differ by "
            f"{worst:.3g}")
    return float(worst)


@torch.no_grad()
def predict(model, tok, texts: list[str], cfg: dict, batch_size: int = 8) -> np.ndarray:
    """Decision scores (higher = more likely present)."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        ids, attn, post = build_inputs(tok, texts[i : i + batch_size], cfg["max_len"])
        out.append(model(ids.cuda(), attn.cuda(), post.cuda()).float().cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------- checkpoints
#
# Stage 2 audits the *trained* weights, so the checkpoint has to be the model
# actually under test: a plain backbone plus the readout it was trained with --
# a head, and for an attention arm the learned query as well. The pooling is
# written into checkpoint.json so a reload rebuilds the same model rather than
# the default one.


def save_checkpoint(model, tok, path):
    """Write a trained arm to `path/` (backbone weights + head + sidecar)."""
    import json
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    backbone = model.backbone
    backbone.save_pretrained(path / "backbone")
    tok.save_pretrained(path / "backbone")
    head = {"head.weight": model.head.weight.detach().cpu(),
            "head.bias": model.head.bias.detach().cpu()}
    # The learned query is part of the readout, not a training artifact: a
    # checkpoint without it would reload as a DIFFERENT model that still runs.
    if hasattr(model, "query"):
        head["query"] = model.query.detach().cpu()
    torch.save(head, path / "head.pt")
    with open(path / "checkpoint.json", "w") as f:
        json.dump({"hidden_size": model.hidden_size,
                   "pooling": getattr(model, "pooling", "mean"),
                   # Load-bearing. A mid-layer checkpoint reloaded without this
                   # would read `last_hidden_state` of a TRUNCATED stack -- a
                   # different function, normed and one site away, that runs
                   # and scores and is silently not the model Stage 2 audits.
                   "head_layer": getattr(model, "head_layer", None)},
                  f, indent=2)
    return path


def load_checkpoint(path, model_name: str | None = None):
    """Reload a saved arm for inference. Returns (model, tokenizer)."""
    import json
    from pathlib import Path

    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    with open(path / "checkpoint.json") as f:
        meta = json.load(f)
    # Must match what training used, or a reloaded checkpoint would score with
    # a different attention function than the one it was fitted under.
    backbone = AutoModel.from_pretrained(
        path / "backbone", dtype=torch.bfloat16,
        attn_implementation=attn_implementation_for(path / "backbone"))
    backbone.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(path / "backbone")
    # Checkpoints written before the attention arm existed carry no `pooling`
    # key and are mean-pooled by definition.
    pooling = meta.get("pooling", "mean")
    # Checkpoints written before the mid-layer arms existed carry no
    # `head_layer` key and read the final layer by definition. The saved
    # backbone is already truncated, so the index is applied but not re-cut.
    head_layer = meta.get("head_layer")
    model = build_classifier(backbone, meta["hidden_size"], pooling, {},
                             head_layer)
    head = torch.load(path / "head.pt", map_location="cpu")
    with torch.no_grad():
        model.head.weight.copy_(head["head.weight"])
        model.head.bias.copy_(head["head.bias"])
        if pooling == "attention":
            if "query" not in head:
                raise KeyError(
                    f"{path} says pooling={pooling!r} but its head.pt has no "
                    f"query; the checkpoint is incomplete")
            model.query.copy_(head["query"])
    model.eval()
    return cuda_or_exit(model, what=f"checkpoint {path.name}"), tok
