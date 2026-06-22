import torch

from src.loss import scale_invariant_loss


def test_loss_invariant_to_global_scale_linear():
    torch.manual_seed(0)
    target = torch.rand(2, 1, 16, 16) + 0.5
    pred = torch.rand(2, 1, 16, 16) + 0.5

    base = scale_invariant_loss(pred, target, lam=1.0)
    for scale in [0.1, 2.0, 5.0, 100.0]:
        scaled = scale_invariant_loss(pred * scale, target, lam=1.0)
        assert torch.allclose(base, scaled, atol=1e-4)


def test_loss_invariant_to_global_scale_log_space():
    torch.manual_seed(1)
    log_target = torch.randn(4, 1, 8, 8)
    log_pred = torch.randn(4, 1, 8, 8)

    base = scale_invariant_loss(log_pred, log_target, lam=1.0, in_log_space=True)
    # A global scale in linear space is a constant offset in log space.
    for offset in [-2.0, 0.5, 3.0]:
        shifted = scale_invariant_loss(
            log_pred + offset, log_target, lam=1.0, in_log_space=True
        )
        assert torch.allclose(base, shifted, atol=1e-5)


def test_loss_zero_when_prediction_matches_up_to_scale():
    torch.manual_seed(2)
    target = torch.rand(2, 1, 8, 8) + 0.5
    # Prediction equals target times a constant: loss should be essentially zero.
    pred = target * 3.7
    loss = scale_invariant_loss(pred, target, lam=1.0)
    assert loss.item() < 1e-6


def test_loss_nonnegative_and_scalar():
    torch.manual_seed(3)
    target = torch.rand(2, 1, 8, 8) + 0.5
    pred = torch.rand(2, 1, 8, 8) + 0.5
    loss = scale_invariant_loss(pred, target, lam=1.0)
    assert loss.dim() == 0
    # With lam = 1 the loss is a variance and so is non negative.
    assert loss.item() >= -1e-7


def test_loss_penalises_non_scale_differences():
    torch.manual_seed(4)
    target = torch.rand(2, 1, 8, 8) + 0.5
    # Random unrelated prediction should give a strictly positive loss.
    pred = torch.rand(2, 1, 8, 8) + 0.5
    loss = scale_invariant_loss(pred, target, lam=1.0)
    assert loss.item() > 1e-4
