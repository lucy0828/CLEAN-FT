"""MentalBERT: the domain-pretrained encoder baseline.

Not run directly — src/stage1/train.py drives it.

MentalBERT is the named exemplar of the paradigm the study critiques
("Transformer Base / Full Fine-Tuning"), and clinical-NLP work on mental-health
text is largely built on BERT-class encoders, so the study should not critique
that paradigm without running it.

It is a DETECTION REFERENCE ARM, not one of the FT arms, and the distinction is
load-bearing. The factorial's entire identification rests on pooling and head
initialization being the only things that vary inside one backbone; MentalBERT
changes backbone, parameter count, pretraining corpus and — decisively — how the
construct reaches the model, since an encoder has no chat template and therefore
no prompting or rubric-carrying probe. Putting it in the factorial would destroy
that identification. It occupies exactly the slot GloVe+BiGRU already occupies:
a published baseline the trained arms are measured against.

This is deliberately a separate module rather than an `is_encoder` flag threaded
through src/baselines/fine_tune.py. The two share almost nothing — no chat
template, no template-token masking — and fine_tune carries the arms
whose comparability the whole study depends on, so it is not worth destabilizing
for a reference arm.

The recipe is the standard BERT fine-tuning recipe from the literature
(configs/default.yaml: encoder_finetune), not the 8B arms' recipe: at lr 1e-5
for 600 steps a 110M encoder is undertrained, and a handicapped baseline would
make the comparison flattering rather than informative.
"""

import numpy as np
import torch
import torch.nn as nn

from src.utils.common import cuda_or_exit

MENTALBERT_REGISTRY = {
    "mentalbert": "mental/mental-bert-base-uncased",
    "mentalroberta": "mental/mental-roberta-base",
    "bert-base": "bert-base-uncased",
}


class MentalBERTClassifier(nn.Module):
    """Encoder -> mean over real tokens -> linear head.

    Mean pooling over the attention mask rather than the [CLS] vector: it is the
    same read-out the decoder arms' mean-pooled head uses, so the one thing this
    baseline does NOT differ in is how the sequence is aggregated.
    """

    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled.float()).squeeze(-1)


def load_mentalbert(model_name: str, seed: int):
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    model_id = MENTALBERT_REGISTRY[model_name]
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModel.from_pretrained(model_id)
    model = MentalBERTClassifier(base, base.config.hidden_size)
    return (cuda_or_exit(model, what="MentalBERT"), tok,
            base.config.max_position_embeddings)


def _batch(tok, texts, max_len):
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True,
              max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


def train(model_name: str, texts: list[str], labels: np.ndarray, cfg: dict,
          seed: int, log_every: int = 0):
    """Fine-tune the encoder on post labels. Returns (model, tokenizer, info)."""
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    model, tok, max_pos = load_mentalbert(model_name, seed)
    max_len = min(cfg["max_len"], max_pos)

    per_epoch = max(1, int(np.ceil(len(texts) / cfg["batch_size"])))
    steps = int(np.clip(per_epoch * cfg["epochs"], cfg["min_steps"],
                        cfg["max_steps"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(
        opt, int(cfg["warmup_ratio"] * steps), steps)
    lossf = nn.BCEWithLogitsLoss()
    y = torch.tensor(labels, dtype=torch.float)
    rng = np.random.default_rng(seed)
    order: list[int] = []

    model.train()
    for step in range(steps):
        if len(order) < cfg["batch_size"]:
            order += rng.permutation(len(texts)).tolist()
        idx, order = order[: cfg["batch_size"]], order[cfg["batch_size"]:]
        ids, attn = _batch(tok, [texts[i] for i in idx], max_len)
        loss = lossf(model(ids.cuda(), attn.cuda()), y[idx].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if log_every and step % log_every == 0:
            print(f"    step {step}/{steps} loss {loss.item():.4f}", flush=True)

    model.eval()
    return model, tok, {"steps": steps, "lr": cfg["lr"], "max_len": max_len,
                        "model_id": MENTALBERT_REGISTRY[model_name],
                        "n_train": len(texts),
                        "params_m": round(sum(p.numel() for p in
                                              model.backbone.parameters()) / 1e6)}


@torch.no_grad()
def predict(model, tok, texts: list[str], cfg: dict, max_len: int,
            batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        ids, attn = _batch(tok, texts[i: i + batch_size], max_len)
        out.append(model(ids.cuda(), attn.cuda()).float().cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(model, encoder: str, max_len: int, path) -> None:
    """The classifier's weights plus what rebuilds it (`load_checkpoint`)."""
    import json
    from pathlib import Path

    path = Path(path)
    torch.save(model.state_dict(), path / "model.pt")
    with open(path / "checkpoint.json", "w") as fh:
        json.dump({"encoder": encoder, "max_len": int(max_len)}, fh)


def load_checkpoint(path):
    """(model, tokenizer, max_len) from `save_checkpoint`."""
    import json
    from pathlib import Path

    path = Path(path)
    with open(path / "checkpoint.json") as fh:
        meta = json.load(fh)
    model, tok, _ = load_mentalbert(meta["encoder"], seed=0)
    state = torch.load(path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tok, int(meta["max_len"])
