import torch

import common as C


def test_tokenize_lowercases_and_drops_html_breaks():
    assert C.tokenize("Great<br />movie!") == ["great", "movie", "!"]


def test_seq_regressor_outputs_one_value_per_sequence():
    for kind in ("rnn", "lstm", "gru"):
        m = C.SeqRegressor(kind, hidden=8, layers=2)
        assert m(torch.randn(4, 30, 1)).shape == (4,)


def test_text_classifier_handles_padding_via_lengths():
    for kind, bidir in (("rnn", False), ("lstm", False), ("lstm", True)):
        m = C.RNNTextClassifier(kind, vocab_size=50, emb=8, hidden=8, bidir=bidir)
        x = torch.randint(1, 50, (3, 12))
        lengths = torch.tensor([12, 7, 1])
        x[1, 7:] = 0
        x[2, 1:] = 0
        assert m(x, lengths).shape == (3, 2)


def test_class_labels():
    assert C.LABELS == ["negative", "positive"]
