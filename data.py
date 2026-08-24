"""Data loading, effective-P sampling, and training-graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np


def _read_split(path: Path) -> Dict[int, Tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: Dict[int, Tuple[int, ...]] = {}
    seen_users = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            tokens = raw.strip().split()
            if not tokens:
                continue
            try:
                user = int(tokens[0])
                items = tuple(int(token) for token in tokens[1:])
            except ValueError as exc:
                raise ValueError(
                    "{}:{} contains a non-integer id".format(path, line_number)
                ) from exc
            if user < 0 or any(item < 0 for item in items):
                raise ValueError("{}:{} contains a negative id".format(path, line_number))
            if user in seen_users:
                raise ValueError("{} contains duplicate user {}".format(path, user))
            seen_users.add(user)
            if len(items) != len(set(items)):
                raise ValueError(
                    "{}:{} contains duplicate item ids".format(path, line_number)
                )
            if items:
                rows[user] = items
    return rows


def _interaction_count(split: Mapping[int, Sequence[int]]) -> int:
    return sum(len(items) for items in split.values())


@dataclass(frozen=True)
class PositiveGroup:
    rows: np.ndarray
    positives: np.ndarray


@dataclass(frozen=True)
class InteractionDataset:
    """Three disjoint implicit-feedback splits with zero-based integer ids."""

    train: Mapping[int, Tuple[int, ...]]
    valid: Mapping[int, Tuple[int, ...]]
    test: Mapping[int, Tuple[int, ...]]
    num_users: int
    num_items: int

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "InteractionDataset":
        root = Path(directory)
        train = _read_split(root / "train_data.txt")
        valid = _read_split(root / "valid_data.txt")
        test = _read_split(root / "test.txt")
        if not train:
            raise ValueError("Training split is empty")

        all_users = set(train) | set(valid) | set(test)
        all_items = {
            item
            for split in (train, valid, test)
            for items in split.values()
            for item in items
        }
        if not all_users or not all_items:
            raise ValueError("Dataset contains no users or items")
        dataset = cls(
            train=train,
            valid=valid,
            test=test,
            num_users=max(all_users) + 1,
            num_items=max(all_items) + 1,
        )
        dataset.validate()
        return dataset

    @property
    def active_users(self) -> np.ndarray:
        return np.asarray(sorted(self.train), dtype=np.int64)

    @property
    def train_interactions(self) -> int:
        return _interaction_count(self.train)

    def validate(self) -> None:
        for split_name, split in (("valid", self.valid), ("test", self.test)):
            missing = sorted(set(split) - set(self.train))
            if missing:
                raise ValueError(
                    "{} has users with no training history: {}".format(
                        split_name, missing[:10]
                    )
                )

        for user in set(self.train) | set(self.valid) | set(self.test):
            train_items = set(self.train.get(user, ()))
            valid_items = set(self.valid.get(user, ()))
            test_items = set(self.test.get(user, ()))
            if train_items & valid_items or train_items & test_items or valid_items & test_items:
                raise ValueError("Split leakage detected for user {}".format(user))

    def effective_p_audit(self, nominal_p: int) -> Dict[str, float]:
        if nominal_p < 1:
            raise ValueError("P must be positive")
        values = np.asarray(
            [min(nominal_p, len(self.train[int(user)])) for user in self.active_users],
            dtype=np.int64,
        )
        return {
            "nominal_p": int(nominal_p),
            "mean": float(values.mean()),
            "min": int(values.min()),
            "max": int(values.max()),
            "users_saturated_below_nominal_p": int((values < nominal_p).sum()),
        }

    def sample_positive_groups(
        self,
        users: Union[Sequence[int], np.ndarray],
        nominal_p: int,
        rng: np.random.Generator,
    ) -> Dict[int, PositiveGroup]:
        if nominal_p < 1:
            raise ValueError("P must be positive")
        pending: Dict[int, Dict[str, List[object]]] = {}
        for row, user_value in enumerate(users):
            user = int(user_value)
            items = np.asarray(self.train[user], dtype=np.int64)
            if items.size < 1:
                raise ValueError("User {} has no training positive".format(user))
            effective_p = min(int(nominal_p), int(items.size))
            chosen = rng.choice(items, size=effective_p, replace=False)
            entry = pending.setdefault(effective_p, {"rows": [], "positives": []})
            entry["rows"].append(row)
            entry["positives"].append(chosen)

        groups: Dict[int, PositiveGroup] = {}
        for effective_p, entry in pending.items():
            row_array = np.asarray(entry["rows"], dtype=np.int64)
            positive_array = np.stack(entry["positives"], axis=0).astype(np.int64)
            if positive_array.shape != (len(row_array), effective_p):
                raise RuntimeError("Effective-P group has an invalid shape")
            groups[effective_p] = PositiveGroup(row_array, positive_array)
        if sum(len(group.rows) for group in groups.values()) != len(users):
            raise RuntimeError("Effective-P groups do not cover the batch")
        return groups

    def normalized_graph(self, device):
        """Build symmetric D^-1/2 A D^-1/2 from training edges only."""
        import torch

        users: List[int] = []
        items: List[int] = []
        for user, positives in self.train.items():
            users.extend([user] * len(positives))
            items.extend(positives)
        user_tensor = torch.as_tensor(users, dtype=torch.long)
        item_tensor = torch.as_tensor(items, dtype=torch.long) + self.num_users
        rows = torch.cat((user_tensor, item_tensor))
        cols = torch.cat((item_tensor, user_tensor))
        node_count = self.num_users + self.num_items
        degree = torch.bincount(rows, minlength=node_count).float()
        inverse_sqrt = degree.clamp_min(1.0).pow(-0.5)
        values = inverse_sqrt[rows] * inverse_sqrt[cols]
        graph = torch.sparse_coo_tensor(
            torch.stack((rows, cols)), values, (node_count, node_count)
        ).coalesce()
        expected_edges = 2 * self.train_interactions
        if graph._nnz() != expected_edges:
            raise ValueError(
                "Training graph has {} edges, expected {}".format(
                    graph._nnz(), expected_edges
                )
            )
        return graph.to(device)
