"""Unit tests for the from-scratch Raschka GPTModel and FunnelDataset."""

import torch
import torch.nn.functional as F

from src.dataset import FunnelDataset, create_dataloader
from src.model import GPTModel, MINI_GPT_CONFIG, MultiHeadAttention


def _make_model(vocab_size: int = 24, context_length: int = 32) -> GPTModel:
    cfg = dict(MINI_GPT_CONFIG)
    cfg["vocab_size"] = vocab_size
    cfg["context_length"] = context_length
    cfg["drop_rate"] = 0.0
    return GPTModel(cfg)


def test_causal_attention_mask_shape_and_values() -> None:
    mha = MultiHeadAttention(
        d_in=32,
        d_out=32,
        context_length=16,
        dropout=0.0,
        num_heads=4,
        qkv_bias=False,
    )
    assert mha.mask.shape == (16, 16)
    # Upper triangle (strictly above diagonal) must be 1, lower triangle + diagonal must be 0
    assert torch.equal(mha.mask, torch.triu(torch.ones(16, 16), diagonal=1))


def test_forward_pass_logits_shape_and_causality() -> None:
    torch.manual_seed(0)
    model = _make_model(vocab_size=24, context_length=32)
    model.eval()

    x = torch.randint(0, 24, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 24)

    # Causality check: changing token at position 8 must NOT affect logits at positions 0..7
    x_perturbed = x.clone()
    x_perturbed[:, 8] = (x_perturbed[:, 8] + 1) % 24
    logits_perturbed = model(x_perturbed)
    assert torch.allclose(logits[:, :8, :], logits_perturbed[:, :8, :], atol=1e-5)


def test_cross_entropy_loss_ignores_padding() -> None:
    torch.manual_seed(0)
    pad_id = 0
    model = _make_model(vocab_size=20, context_length=16)
    model.eval()

    inputs = torch.tensor([[2, 4, 9, 5, 10, 0, 0]], dtype=torch.long)
    targets_a = torch.tensor([[4, 9, 5, 10, 3, 0, 0]], dtype=torch.long)
    targets_b = torch.tensor([[4, 9, 5, 10, 3, 0, 0]], dtype=torch.long)

    logits = model(inputs)
    loss_padded = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets_a.view(-1),
        ignore_index=pad_id,
    )
    loss_unpadded = F.cross_entropy(
        logits[:, :5, :].reshape(-1, logits.size(-1)),
        targets_b[:, :5].reshape(-1),
        ignore_index=pad_id,
    )
    assert torch.isfinite(loss_padded)
    assert torch.allclose(loss_padded, loss_unpadded, atol=1e-6)


def test_sliding_window_dataset_and_dataloader() -> None:
    short_seq = [2, 4, 9, 5, 10, 3]  # len 6 -> 5 input/target tokens
    long_seq = list(range(2, 2 + 25))  # len 25 -> sliding windows of length 8
    ds = FunnelDataset([short_seq, long_seq], context_length=8, stride=4)
    assert len(ds) >= 5

    loader = create_dataloader([short_seq, long_seq], context_length=8, pad_id=0, batch_size=2, shuffle=False)
    inp, tgt = next(iter(loader))
    assert inp.shape == tgt.shape
    assert inp.shape[0] == 2
