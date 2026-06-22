"""Training helpers for the depth network.

These functions are intentionally small so they can be exercised by tests and
also run as a short demo from the command line.
"""

import torch

from .model import DepthNet
from .loss import scale_invariant_loss
from .data import make_synthetic_batch


def train_on_batch(model, images, depths, steps=50, lr=1e-3, lam=1.0):
    """Run a few optimisation steps on a single fixed batch.

    Returns a list with the loss value before and at every step, so callers can
    check that the loss went down.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        log_pred = model(images)
        log_target = torch.log(depths + 1e-6)
        loss = scale_invariant_loss(log_pred, log_target, lam=lam, in_log_space=True)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


def main():
    torch.manual_seed(0)
    images, depths = make_synthetic_batch(batch_size=8, height=32, width=32, seed=0)
    model = DepthNet()
    history = train_on_batch(model, images, depths, steps=100, lr=1e-3)
    print(f"start loss {history[0]:.4f}  end loss {history[-1]:.4f}")


if __name__ == "__main__":
    main()
