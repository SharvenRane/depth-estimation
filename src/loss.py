"""Scale invariant depth loss.

This follows the formulation from Eigen, Puhrsch and Fergus (2014). Working in
log space, the loss compares the difference between prediction and target while
subtracting out any constant offset. A constant offset in log space corresponds
to a global multiplicative scale in linear space, so the loss is invariant to
multiplying the predicted depth map by any positive scalar.

For a per pixel log difference d_i = log(pred_i) - log(target_i) over N valid
pixels the loss is:

    L = mean(d_i^2) - lam * mean(d_i)^2

With lam = 1 the loss depends only on the variance of d_i and so is fully
invariant to a constant offset in d_i. The default lam = 0.85 keeps most of the
scale invariance while still penalising large absolute errors a little, which is
the value commonly used in the literature.
"""

import torch


def scale_invariant_loss(pred, target, lam=1.0, eps=1e-6, in_log_space=False):
    """Compute the scale invariant log loss between pred and target.

    Args:
        pred: predicted depth, shape (..., H, W). If ``in_log_space`` is False
            these are linear depths and must be positive; the log is taken
            internally with a small epsilon for numerical safety.
        target: ground truth depth, same shape as pred.
        lam: weight on the squared mean term. lam = 1 gives full scale
            invariance. Values in [0, 1] are valid.
        eps: small constant added before taking logs of linear depth.
        in_log_space: if True, pred and target are already log depths and are
            used directly.

    Returns:
        A scalar tensor with the mean loss over the batch.
    """
    if in_log_space:
        log_pred = pred
        log_target = target
    else:
        log_pred = torch.log(pred + eps)
        log_target = torch.log(target + eps)

    d = log_pred - log_target
    # Flatten everything except a possible leading batch dimension is not needed
    # here; we reduce over all elements to get a single scalar.
    n = d.numel()
    term_mse = torch.sum(d * d) / n
    term_offset = (torch.sum(d) / n) ** 2
    return term_mse - lam * term_offset
