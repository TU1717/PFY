"""Full-catalog ranking metrics for implicit-feedback recommendation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from data import InteractionDataset


try:
    import torch
except ModuleNotFoundError:  # Pure metric tests do not require PyTorch.
    torch = None


def metrics_for_user(
    recommendations: Sequence[int],
    relevant_items: Sequence[int],
    cutoffs: Sequence[int],
) -> Dict[str, float]:
    """Calculate one user's contributions to the reported metrics."""
    relevant = set(int(item) for item in relevant_items)
    if not relevant:
        raise ValueError("A metric user must have at least one relevant item")
    ranked = np.asarray(recommendations, dtype=np.int64)
    output: Dict[str, float] = {}
    for cutoff in cutoffs:
        predicted = ranked[:cutoff]
        hits = np.asarray(
            [1.0 if int(item) in relevant else 0.0 for item in predicted],
            dtype=np.float64,
        )
        hit_count = float(hits.sum())
        output["precision@{}".format(cutoff)] = hit_count / float(cutoff)
        output["recall@{}".format(cutoff)] = hit_count / float(len(relevant))

        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        dcg = float((hits * discounts).sum())
        ideal_length = min(cutoff, len(relevant))
        idcg = float(discounts[:ideal_length].sum())
        output["ndcg@{}".format(cutoff)] = dcg / idcg if idcg else 0.0

        reciprocal = 1.0 / np.arange(1, cutoff + 1, dtype=np.float64)
        output["mrr_numerator@{}".format(cutoff)] = float(
            (hits * reciprocal).sum()
        )
    return output


def evaluate(
    model: object,
    dataset: InteractionDataset,
    split: str,
    cutoffs: Sequence[int] = (5, 10, 20),
    batch_size: int = 512,
) -> Dict[str, float]:
    """Rank every catalog item after masking the user's training history."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for full-catalog evaluation")
    with torch.no_grad():
        return _evaluate(model, dataset, split, cutoffs, batch_size)


def _evaluate(
    model: object,
    dataset: InteractionDataset,
    split: str,
    cutoffs: Sequence[int],
    batch_size: int,
) -> Dict[str, float]:
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    ground_truth: Mapping[int, Sequence[int]] = getattr(dataset, split)
    if not ground_truth:
        raise ValueError("{} split is empty".format(split))
    cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError("cutoffs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    max_k = max(cutoffs)
    if max_k > dataset.num_items:
        raise ValueError("largest cutoff exceeds catalog size")

    model.eval()
    all_user, all_item = model.compute()
    device = all_user.device
    users = sorted(ground_truth)
    insufficient = [
        user
        for user in users
        if dataset.num_items - len(dataset.train.get(user, ())) < max_k
    ]
    if insufficient:
        raise ValueError(
            "{} evaluation users have fewer than {} unobserved items".format(
                len(insufficient), max_k
            )
        )

    metric_names = [
        "{}@{}".format(name, cutoff)
        for cutoff in cutoffs
        for name in ("precision", "recall", "ndcg")
    ]
    totals = {name: 0.0 for name in metric_names}
    mrr_totals = {cutoff: 0.0 for cutoff in cutoffs}
    total_relevant = sum(len(ground_truth[user]) for user in users)

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        user_tensor = torch.as_tensor(batch_users, dtype=torch.long, device=device)
        scores = all_user[user_tensor] @ all_item.t()
        for row, user in enumerate(batch_users):
            train_items = dataset.train.get(user, ())
            if train_items:
                item_tensor = torch.as_tensor(
                    train_items, dtype=torch.long, device=device
                )
                scores[row, item_tensor] = -torch.inf
        recommendations = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()

        for row, user in enumerate(batch_users):
            contributions = metrics_for_user(
                recommendations[row], ground_truth[user], cutoffs
            )
            for name in metric_names:
                totals[name] += contributions[name]
            for cutoff in cutoffs:
                mrr_totals[cutoff] += contributions[
                    "mrr_numerator@{}".format(cutoff)
                ]

    user_count = float(len(users))
    output = {name: totals[name] / user_count for name in metric_names}
    for cutoff in cutoffs:
        output["mrr@{}".format(cutoff)] = (
            mrr_totals[cutoff] / float(total_relevant)
        )
    return output
