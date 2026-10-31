"""GloVe + BiGRU: the baseline family the corpus's own paper used.

How to run (GPU; falls back to CPU when no card is visible):
    uv run python -m src.stage1.train --method glove_bigru

Why this arm exists. LonXplain (Garg et al., 2023) evaluates linear classifiers
over Word2Vec and GloVe vectors fed to LSTM/GRU recurrent encoders. Every
number in this study is measured against decoder LLMs, so a reader who knows
the corpus cannot tell how much the backbone buys over the baseline the
dataset shipped with. This arm answers that in the study's own protocol: same
training subsamples, same detection metrics.

Out-of-vocabulary handling: in a mean-pooled setup an OOV word is best
*omitted*, because a zero vector is a location in the space and would drag the
post vector. A recurrent encoder reads a sequence, so omitting a word would
silently delete it from the post and misalign every word index against the gold
span. Here OOV words are kept in place and given a zero vector, which the GRU
is free to learn to ignore.
"""

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.utils.data_prep import words

MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"

_TABLE: tuple[dict, np.ndarray] | None = None


def device() -> torch.device:
    """The card the BiGRU trains on, or CPU when no GPU is visible.

    The net is small but its sequences are not: at max_len 512 and batch 16 a
    CPU walks 512 timesteps a step at a time, where a fused cuDNN GRU does the
    same fold about 19x faster — hours against minutes over the ten folds.
    `--devices cpu` empties CUDA_VISIBLE_DEVICES and this falls back to CPU;
    the fit is seeded either way, so the card is a speed choice, not a
    protocol one.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_table() -> tuple[dict, np.ndarray]:
    """The 400k GloVe vocabulary and its embedding matrix, loaded once.

    Read straight off the embedding layer rather than through `encode()`,
    because the packaged tokenizer applies sklearn's 386-word stopword list,
    which contains "alone" — dropping the most on-topic word in this corpus
    would cripple the baseline in our own favour.
    """
    global _TABLE
    if _TABLE is None:
        # device="cpu" here is about the table read, not about where this
        # arm trains. SentenceTransformer defaults to CUDA when a card is
        # visible, and the 400k x 300 matrix is read once and copied straight
        # to host, so staging it on the card is a round trip that buys
        # nothing. The BiGRU itself trains on `device()`.
        enc = SentenceTransformer(MODEL, device="cpu")
        raw = enc[0].tokenizer.get_vocab()
        vocab = raw if isinstance(raw, dict) else {w: i for i, w in enumerate(raw)}
        _TABLE = (vocab, enc[0].emb_layer.weight.detach().cpu().numpy())
    return _TABLE


def encode_words(text: str, max_len: int) -> np.ndarray:
    """One GloVe row per word of `text`, in order, zero-filled where OOV."""
    vocab, emb = embedding_table()
    post_words = words(text)[:max_len]
    out = np.zeros((len(post_words), emb.shape[1]), dtype=np.float32)
    for i, w in enumerate(post_words):
        idx = vocab.get(w["norm"])
        if idx is not None:
            out[i] = emb[idx]
    return out


def _batch(texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [encode_words(t, max_len) for t in texts]
    lens = [max(len(s), 1) for s in seqs]
    width, dim = max(lens), embedding_table()[1].shape[1]
    x = torch.zeros(len(seqs), width, dim)
    mask = torch.zeros(len(seqs), width)
    for i, s in enumerate(seqs):
        if len(s):
            x[i, : len(s)] = torch.from_numpy(s)
            mask[i, : len(s)] = 1.0
    return x, mask


class BiGRUClassifier(nn.Module):
    """BiGRU over frozen GloVe vectors, mean-pooled over real words.

    Embeddings stay frozen: the point of the arm is what a recurrent encoder
    extracts from *static* vectors, and at n=25 a trainable 400k-row embedding
    matrix would be fitting vocabulary, not language.
    """

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        # Mean over real words only, so padding cannot dilute a short post.
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(self.drop(pooled)).squeeze(-1)


def train(texts: list[str], labels: np.ndarray, cfg: dict, seed: int = 0):
    """Fit one BiGRU cell. Returns the model in eval mode."""
    torch.manual_seed(seed)
    gcfg = cfg["glove_bigru"]
    dim = embedding_table()[1].shape[1]
    dev = device()
    model = BiGRUClassifier(dim, gcfg["hidden"], gcfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=gcfg["lr"],
                            weight_decay=gcfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    y = torch.from_numpy(labels.astype(np.float32)).to(dev)
    bs, max_len = gcfg["batch_size"], gcfg["max_len"]

    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(gcfg["epochs"]):
        order = rng.permutation(len(texts))
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            # Word lookup stays on the host — it is a numpy gather over the
            # GloVe table — and only the padded batch crosses to the card.
            x, mask = _batch([texts[j] for j in idx], max_len)
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev), mask.to(dev)), y[idx])
            loss.backward()
            # Recurrent nets at n=25 are the classic exploding-gradient case;
            # clipping is what makes the scarce end of the axis trainable.
            nn.utils.clip_grad_norm_(model.parameters(), gcfg["clip"])
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, texts: list[str], cfg: dict, batch_size: int = 64) -> np.ndarray:
    """Decision scores (logits) for `texts`, matching the fine-tune arm's contract."""
    max_len = cfg["glove_bigru"]["max_len"]
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        x, mask = _batch(texts[i : i + batch_size], max_len)
        out.append(model(x.to(dev), mask.to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def save_checkpoint(model, cfg: dict, path) -> None:
    torch.save(model.state_dict(), __import__("pathlib").Path(path) / "model.pt")


def load_checkpoint(path, cfg: dict):
    """The BiGRU rebuilt from config and its saved weights, in eval mode."""
    gcfg = cfg["glove_bigru"]
    dim = embedding_table()[1].shape[1]
    model = BiGRUClassifier(dim, gcfg["hidden"], gcfg["dropout"]).to(device())
    model.load_state_dict(torch.load(
        __import__("pathlib").Path(path) / "model.pt", map_location=device()))
    model.eval()
    return model
