from __future__ import annotations

import torch


def l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the L1 loss between input and target tensors.

    Args:
        input: The predicted values.
        target: The ground truth values.

    Returns:
        The computed L1 loss.
    """

    return torch.mean(torch.abs(input - target))


def l2_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 loss (mean squared error) between input and target tensors.

    Args:
        input: The predicted values.
        target: The ground truth values.

    Returns:
        The computed L2 loss.
    """

    return torch.mean((input - target) ** 2)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Computes the symmetric Kullback-Leibler divergence between two distributions
    using torch.nn.functional.kl_div.

    Args:
        p: The true probability distribution.
        q: The predicted probability distribution.

    Returns:
        The computed symmetric KL divergence.
    """

    loss_func = torch.nn.KLDivLoss()

    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)

    kl_pq = loss_func(p.log(), q)
    kl_qp = loss_func(q.log(), p)
    symmetric_kl = kl_pq + kl_qp

    return symmetric_kl.mean()
