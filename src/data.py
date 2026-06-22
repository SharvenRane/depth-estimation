"""Synthetic image and depth map generator.

Each sample is a smooth depth map built from a small number of random Gaussian
blobs on a sloped background. The matching image is a deterministic shading of
that depth map plus a little noise, so there is a real signal connecting image
appearance to depth that a small network can learn.
"""

import torch


def _gaussian_blob(h, w, cx, cy, sigma, device):
    ys = torch.arange(h, device=device).view(h, 1).float()
    xs = torch.arange(w, device=device).view(1, w).float()
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def make_synthetic_batch(batch_size=4, height=32, width=32, n_blobs=3, seed=None,
                         device="cpu"):
    """Create a batch of synthetic images and matching depth maps.

    Args:
        batch_size: number of samples.
        height, width: spatial size.
        n_blobs: number of Gaussian blobs added per depth map.
        seed: optional int for reproducibility.
        device: torch device string.

    Returns:
        images: tensor of shape (B, 3, H, W) with values roughly in [0, 1].
        depths: tensor of shape (B, 1, H, W) with strictly positive values.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=device)

    images = torch.zeros(batch_size, 3, height, width, device=device)
    depths = torch.zeros(batch_size, 1, height, width, device=device)

    ys = torch.linspace(0, 1, height, device=device).view(height, 1)
    xs = torch.linspace(0, 1, width, device=device).view(1, width)

    for b in range(batch_size):
        # Sloped background plane so depth varies smoothly across the frame.
        slope_y = (rand(1).item() - 0.5) * 1.5
        slope_x = (rand(1).item() - 0.5) * 1.5
        depth = 1.0 + 0.3 * (slope_y * ys + slope_x * xs)

        for _ in range(n_blobs):
            cx = rand(1).item() * width
            cy = rand(1).item() * height
            sigma = 3.0 + rand(1).item() * (min(height, width) / 6.0)
            amp = 0.5 + rand(1).item()
            depth = depth + amp * _gaussian_blob(height, width, cx, cy, sigma, device)

        # Keep depth strictly positive.
        depth = depth - depth.min() + 0.5
        depths[b, 0] = depth

        # Image shading: a deterministic function of depth across channels plus
        # mild noise. This gives the network a learnable appearance to depth map.
        shade = depth / (depth.max() + 1e-6)
        noise = 0.02 * rand(3, height, width)
        images[b, 0] = shade
        images[b, 1] = 1.0 - shade
        images[b, 2] = torch.sin(3.0 * shade)
        images[b] = images[b] + noise
        images[b] = images[b].clamp(0.0, 1.0)

    return images, depths
