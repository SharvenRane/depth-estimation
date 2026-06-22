import torch

from src.model import DepthNet
from src.data import make_synthetic_batch


def test_forward_output_matches_depth_shape():
    model = DepthNet()
    images, depths = make_synthetic_batch(batch_size=2, height=32, width=32, seed=1)
    out = model(images)
    # One channel depth map at the same spatial resolution as the target.
    assert out.shape == depths.shape
    assert out.shape == (2, 1, 32, 32)


def test_forward_handles_nonsquare_and_other_sizes():
    model = DepthNet()
    images, depths = make_synthetic_batch(batch_size=1, height=48, width=64, seed=2)
    out = model(images)
    assert out.shape == depths.shape == (1, 1, 48, 64)


def test_output_is_finite():
    model = DepthNet()
    images, _ = make_synthetic_batch(batch_size=2, height=32, width=32, seed=3)
    out = model(images)
    assert torch.isfinite(out).all()


def test_data_depth_is_positive():
    _, depths = make_synthetic_batch(batch_size=3, height=32, width=32, seed=4)
    assert (depths > 0).all()
