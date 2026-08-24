"""Full-catalog Gumbel-PFY with an exact hard Top-P oracle."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def sample_gumbel(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw independent standard Gumbel random variables."""
    if not dtype.is_floating_point:
        raise ValueError("Gumbel noise requires a floating-point dtype")
    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    machine_eps = torch.finfo(dtype).eps
    uniform = uniform.clamp(min=machine_eps, max=1.0 - machine_eps)
    return -torch.log(-torch.log(uniform))


def full_catalog_gumbel_pfy_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    p: int,
    mc_samples: int = 32,
    mc_chunk: int = 4,
    gumbel_scale: float = 0.2,
    generator: Optional[torch.Generator] = None,
    fixed_noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the Monte Carlo Gumbel-PFY objective and diagnostics.

    ``scores`` contains one score for every catalog item. The optimized scalar
    is the perturbed Top-P potential minus the sum of the P target scores. The
    target-only Fenchel-Young regularizer is omitted because it is constant
    with respect to the recommender parameters.

    ``fixed_noise`` is reserved for deterministic tests. Normal training draws
    fresh independent Gumbel variables on every call.
    """
    if scores.ndim != 2 or not scores.dtype.is_floating_point:
        raise ValueError("scores must be a floating tensor with shape [B, I]")
    if positive_indices.ndim != 2:
        raise ValueError("positive_indices must have shape [B, P]")

    batch_size, num_items = scores.shape
    p = int(p)
    mc_samples = int(mc_samples)
    mc_chunk = int(mc_chunk)
    gumbel_scale = float(gumbel_scale)

    if not 1 <= p <= num_items:
        raise ValueError("P must be between one and the catalog size")
    if tuple(positive_indices.shape) != (batch_size, p):
        raise ValueError("positive_indices has the wrong shape")
    if mc_samples < 1 or mc_chunk < 1:
        raise ValueError("mc_samples and mc_chunk must be positive")
    if gumbel_scale <= 0.0:
        raise ValueError("gumbel_scale must be positive")

    positive_indices = positive_indices.long()
    if positive_indices.numel():
        lower = int(positive_indices.min().item())
        upper = int(positive_indices.max().item())
        if lower < 0 or upper >= num_items:
            raise ValueError("positive_indices contains an item outside the catalog")
    if p > 1:
        ordered = torch.sort(positive_indices, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise ValueError("Each target row must contain P distinct items")

    noise_shape = (mc_samples, batch_size, num_items)
    if fixed_noise is not None:
        if tuple(fixed_noise.shape) != noise_shape:
            raise ValueError("fixed_noise has the wrong shape")
        if fixed_noise.device != scores.device or fixed_noise.dtype != scores.dtype:
            raise ValueError("fixed_noise must match scores in device and dtype")

    target_score_sum = scores.gather(1, positive_indices).sum(dim=1)
    potential_sum = scores.new_zeros(batch_size)
    target_hit_sum = scores.new_zeros(())

    completed = 0
    while completed < mc_samples:
        chunk_size = min(mc_chunk, mc_samples - completed)
        if fixed_noise is None:
            noise = sample_gumbel(
                (chunk_size, batch_size, num_items),
                device=scores.device,
                dtype=scores.dtype,
                generator=generator,
            )
        else:
            noise = fixed_noise[completed : completed + chunk_size]

        perturbed = scores.unsqueeze(0) + gumbel_scale * noise
        top_values, top_indices = torch.topk(
            perturbed,
            k=p,
            dim=-1,
            largest=True,
            sorted=False,
        )
        potential_sum = potential_sum + top_values.sum(dim=-1).sum(dim=0)

        with torch.no_grad():
            target_hits = (
                top_indices.unsqueeze(-1)
                == positive_indices.unsqueeze(0).unsqueeze(2)
            ).any(dim=-1)
            target_hit_sum = target_hit_sum + target_hits.sum().to(scores.dtype)
        completed += chunk_size

    expected_top_p_value = potential_sum / float(mc_samples)
    loss = (expected_top_p_value - target_score_sum).mean()
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "expected_top_p_value": float(expected_top_p_value.mean().detach().cpu()),
        "target_score_sum": float(target_score_sum.mean().detach().cpu()),
        "expected_target_hits": float(
            (target_hit_sum / float(mc_samples * batch_size)).detach().cpu()
        ),
        "expected_target_recall": float(
            (target_hit_sum / float(mc_samples * batch_size * p)).detach().cpu()
        ),
        "selection_mass": float(p),
        "mc_samples": float(mc_samples),
        "mc_chunk": float(mc_chunk),
        "gumbel_scale": gumbel_scale,
    }
    return loss, diagnostics
