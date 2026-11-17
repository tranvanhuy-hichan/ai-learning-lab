"""Sequence models. Layouts match projects/sequence-modeling/common.py so the saved weights load as-is."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SeqRegressor(nn.Module):
    """30 days of temperature -> RNN/LSTM -> last hidden state -> tomorrow's (normalised) temperature."""

    def __init__(self, kind: str = "rnn", hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(1, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1]).squeeze(-1)


class RNNTextClassifier(nn.Module):
    """Embedding -> RNN/LSTM (padding skipped with pack_padded_sequence) -> final state -> Linear."""

    def __init__(
        self,
        kind: str,
        vocab_size: int,
        emb: int = 128,
        hidden: int = 128,
        layers: int = 1,
        bidir: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.kind, self.bidir = kind, bidir
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = cls(emb, hidden, layers, batch_first=True, bidirectional=bidir, dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.kind == "lstm":
            h = h[0]
        h = torch.cat([h[-2], h[-1]], 1) if self.bidir else h[-1]
        return self.fc(self.drop(h))
