import torch

from src.model import DepthNet
from src.data import make_synthetic_batch
from src.train import train_on_batch


def test_training_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    images, depths = make_synthetic_batch(batch_size=4, height=32, width=32, seed=0)
    model = DepthNet()

    history = train_on_batch(model, images, depths, steps=60, lr=1e-3, lam=1.0)

    assert len(history) == 60
    assert all(torch.isfinite(torch.tensor(h)) for h in history)
    # The end loss should be clearly below the starting loss.
    assert history[-1] < history[0]
    assert history[-1] < 0.5 * history[0]


def test_training_step_changes_parameters():
    torch.manual_seed(1)
    images, depths = make_synthetic_batch(batch_size=2, height=32, width=32, seed=1)
    model = DepthNet()

    before = [p.detach().clone() for p in model.parameters()]
    train_on_batch(model, images, depths, steps=1, lr=1e-3)
    after = list(model.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
