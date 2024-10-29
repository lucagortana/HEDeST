from __future__ import annotations

import torch


def weighted_l1_loss(input, target, weights, reduction="mean"):
    loss = weights * torch.abs(input - target)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


def weighted_l2_loss(input, target, weights, reduction="mean"):
    loss = weights * (input - target) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


def weighted_kl_divergence(p, q, weights, reduction="sum"):
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)

    loss = p * torch.log(p / q) - p + q
    w_loss = weights * loss

    if reduction == "mean":
        return w_loss.mean()
    elif reduction == "sum":
        return w_loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


def shannon_entropy(U):
    return -(U * torch.log(U + 1e-12)).sum()


def ROT(F, z, alpha=0.5, epsilon=1, n_iter=75, weights=None, reduction="sum"):
    """
    Compute a differentiable approximation to f_relax-ent using Algorithm 1.

    Args:
        F (torch.Tensor): Tensor of shape (n, k) representing the predicted probability distributions.
        z (torch.Tensor): Tensor of shape (k,) representing the target distribution.
        alpha (float): Smoothing parameter in [0, 1].
        epsilon (float): Entropy parameter, must be > 0.
        n_iter (int): Number of Sinkhorn iterations.

    Returns:
        torch.Tensor: The differentiable approximation to f_relax-ent(F, z).
    """
    n, _ = F.shape
    # Step 1: Initialize variables
    K = F.pow(1 / epsilon)
    K = K.t()

    tau = (1 + alpha * epsilon / (1 - alpha)) ** -1

    b = torch.ones(n, device=F.device)

    # Steps 2-3: Perform Sinkhorn iterations
    for _ in range(n_iter):
        # Update a
        a = ((n * z / (K @ b)).pow(tau)).float()
        # Update b
        b = (torch.ones(n, device=F.device) / (K.t() @ a)).float()

    # Step 6: Compute U
    U = torch.diag(a) @ K @ torch.diag(b)

    # Step 7: Compute the final approximation to f_relax-ent
    trace_logF_U = torch.trace(torch.log(F) @ U)
    entropy_U = shannon_entropy(U)
    kld = weighted_kl_divergence(U @ torch.ones(n, device=F.device) / n, z, weights=weights, reduction=reduction)

    # Compute the final differentiable approximation
    return (alpha / n) * (-trace_logF_U - epsilon * entropy_U) + (1 - alpha) * kld
