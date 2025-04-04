from __future__ import annotations

import torch

# from typing import Optional


def weighted_l1_loss(
    input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes the weighted L1 loss between input and target tensors.

    Args:
        input (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        weights (torch.Tensor): Per-element weights for the loss.
        reduction (str): Specifies the reduction to apply to the output.
                         Options are 'mean' (default) or 'sum'.

    Returns:
        torch.Tensor: The computed weighted L1 loss. If `reduction` is 'mean', returns the mean loss;
                      if 'sum', returns the sum of the losses.
    """

    loss = weights * torch.abs(input - target)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


def weighted_l2_loss(
    input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes the weighted L2 loss between input and target tensors.

    Args:
        input (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        weights (torch.Tensor): Per-element weights for the loss.
        reduction (str): Specifies the reduction to apply to the output.
                         Options are 'mean' (default) or 'sum'.

    Returns:
        torch.Tensor: The computed weighted L2 loss. If `reduction` is 'mean', returns the mean loss;
                      if 'sum', returns the sum of the losses.
    """

    loss = weights * (input - target) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


def weighted_kl_divergence(
    p: torch.Tensor, q: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes the weighted symmetric Kullback-Leibler divergence between two distributions
    using torch.nn.functional.kl_div.

    Args:
        p (torch.Tensor): The true probability distribution.
        q (torch.Tensor): The predicted probability distribution.
        weights (torch.Tensor): Per-element weights for the divergence.
        reduction (str): Specifies the reduction to apply to the output ('mean' or 'sum').

    Returns:
        torch.Tensor: The computed weighted symmetric KL divergence.
    """
    loss_func = torch.nn.KLDivLoss()

    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)

    kl_pq = loss_func(p.log(), q)
    kl_qp = loss_func(q.log(), p)
    symmetric_kl = kl_pq + kl_qp

    if reduction == "mean":
        return symmetric_kl.mean()
    elif reduction == "sum":
        return symmetric_kl.mean()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


# def weighted_kl_divergence(
#     p: torch.Tensor, q: torch.Tensor, weights: torch.Tensor, reduction: str = "mean"
# ) -> torch.Tensor:
#     """
#     Computes the weighted Kullback-Leibler divergence between two distributions.

#     Args:
#         p (torch.Tensor): The true probability distribution (must be positive).
#         q (torch.Tensor): The predicted probability distribution (must be positive).
#         weights (torch.Tensor): Per-element weights for the divergence.
#         reduction (str): Specifies the reduction to apply to the output.
#                          Options are 'mean' (default) or 'sum'.

#     Returns:
#         torch.Tensor: The computed weighted KL divergence. If `reduction` is 'mean', returns the mean loss;
#                       if 'sum', returns the sum of the losses.
#     """

#     p = p.clamp(min=1e-10)
#     q = q.clamp(min=1e-10)

#     loss = p * torch.log(p / q) - p + q
#     w_loss = weights * loss

#     if reduction == "mean":
#         return w_loss.mean()
#     elif reduction == "sum":
#         return w_loss.sum()
#     else:
#         raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")


# def shannon_entropy(U: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the Shannon entropy of a probability distribution.

#     Args:
#         U (torch.Tensor): A probability distribution tensor.

#     Returns:
#         torch.Tensor: The Shannon entropy of the input distribution.
#     """

#     return -(U * torch.log(U + 1e-12)).sum()


# def ROT(
#     F: torch.Tensor,
#     z: torch.Tensor,
#     alpha: float = 0.5,
#     epsilon: float = 1.0,
#     n_iter: int = 75,
#     weights: Optional[torch.Tensor] = None,
# ) -> tuple:
#     """
#     Computes the Relaxed Optimal Transport (ROT) loss.

#     Args:
#         F (torch.Tensor): Input matrix of size (n, n).
#         z (torch.Tensor): Target distribution of size (n,).
#         alpha (float): Weight for the entropy loss term (default: 0.5).
#         epsilon (float): Regularization parameter (default: 1.0).
#         n_iter (int): Number of Sinkhorn iterations (default: 75).
#         weights (torch.Tensor, optional): Optional weights for KL divergence computation.

#     Returns:
#         tuple: A tuple containing:
#                - loss (torch.Tensor): The total ROT loss.
#                - entropy_loss (torch.Tensor): The entropy-based loss component.
#                - kld (torch.Tensor): The KL divergence loss component.
#     """

#     n, _ = F.shape
#     # Step 1: Initialize variables
#     K = F.pow(1 / epsilon)
#     K = K.t()

#     tau = (1 + alpha * epsilon / (1 - alpha)) ** -1

#     b = torch.ones(n, device=F.device)

#     # Steps 2: Perform Sinkhorn iterations
#     for _ in range(n_iter):
#         # Update a
#         a = (n * z / (K @ b)).pow(tau)
#         # Update b
#         b = torch.ones(n, device=F.device) / (K.t() @ a)

#     # Step 3: Compute U
#     U = torch.diag(a) @ K @ torch.diag(b)

#     # Step 4: Compute the final approximation to f_relax-ent
#     trace_logF_U = torch.trace(torch.log(F) @ U)
#     entropy_U = shannon_entropy(U)
#     kld = weighted_kl_divergence(U @ torch.ones(n, device=F.device) / n, z, weights=weights, reduction="sum")

#     # Compute the final differentiable approximation
#     entropy_loss = -trace_logF_U - epsilon * entropy_U
#     loss = (alpha / n) * entropy_loss + (1 - alpha) * kld
#     return loss, entropy_loss, kld
