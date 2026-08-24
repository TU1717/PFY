"""Training, checkpoint selection, test-once evaluation, and aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from data import InteractionDataset
from losses import full_catalog_gumbel_pfy_loss
from metrics import evaluate
from models import BaseRecommender, build_model


BACKBONES = ("mf", "lightgcn", "xsimgcl")


def parse_integer_list(value: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return parsed


def normalize_integer_list(value: Iterable[int], name: str) -> Tuple[int, ...]:
    if isinstance(value, str):
        result = parse_integer_list(value)
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("{} must not be empty".format(name))
    return result


def build_parser(default_config: str = "config.json") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_integer_list,
        default=(2024, 2025, 2026, 2027, 2028),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train-batch", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--cutoffs", type=parse_integer_list, default=(5, 10, 20))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--lightgcn-layers", type=int, default=2)
    parser.add_argument("--xsimgcl-layers", type=int, default=3)

    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--gumbel-scale", type=float, default=0.2)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--mc-chunk", type=int, default=4)

    parser.add_argument("--cl-rate", type=float, default=0.05)
    parser.add_argument("--xsimgcl-eps", type=float, default=0.1)
    parser.add_argument("--cl-temp", type=float, default=0.1)
    parser.add_argument("--cl-layer", type=int, default=0)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def atomic_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=str(path.parent), delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("Non-finite gradient")
        squared_norm += float(parameter.grad.detach().float().square().sum().cpu())
        found = True
    if not found or squared_norm <= 0.0:
        raise RuntimeError("Missing or zero gradient")
    return math.sqrt(squared_norm)


def scientific_configuration(args: argparse.Namespace) -> Dict[str, object]:
    excluded = {"config", "data_dir", "output_dir", "device"}
    result = {key: value for key, value in vars(args).items() if key not in excluded}
    for key in ("seeds", "cutoffs"):
        if key in result:
            result[key] = [int(item) for item in result[key]]
    return result


def batch_loss(
    args: argparse.Namespace,
    dataset: InteractionDataset,
    model: BaseRecommender,
    users_np: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(model.parameters()).device
    users = torch.as_tensor(users_np, dtype=torch.long, device=device)
    groups = dataset.sample_positive_groups(users_np, args.p, rng)

    all_users, all_items = model.compute()
    full_scores = all_users[users] @ all_items.t()
    ranking_loss = full_scores.new_zeros(())
    positive_anchors = torch.empty(len(users_np), dtype=torch.long, device=device)
    weighted_hits = 0.0
    effective_p_sum = 0.0

    for effective_p, group in sorted(groups.items()):
        rows = torch.as_tensor(group.rows, dtype=torch.long, device=device)
        positives = torch.as_tensor(group.positives, dtype=torch.long, device=device)
        group_scores = full_scores.index_select(0, rows)
        group_loss, group_diagnostics = full_catalog_gumbel_pfy_loss(
            group_scores,
            positives,
            p=effective_p,
            mc_samples=args.mc_samples,
            mc_chunk=args.mc_chunk,
            gumbel_scale=args.gumbel_scale,
        )
        weight = float(rows.numel()) / float(len(users_np))
        ranking_loss = ranking_loss + weight * group_loss
        positive_anchors[rows] = positives[:, 0]
        weighted_hits += weight * group_diagnostics["expected_target_hits"]
        effective_p_sum += float(effective_p * rows.numel())

    effective_p_mean = effective_p_sum / float(len(users_np))
    contrastive_loss = model.contrastive_loss(
        users,
        positive_anchors,
        all_users,
        all_items,
    )
    total = ranking_loss + contrastive_loss
    diagnostics = {
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "effective_p_mean": effective_p_mean,
        "expected_target_hits": weighted_hits,
    }
    return total, diagnostics


def run_seed(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    if 5 not in args.cutoffs:
        raise ValueError("cutoffs must include 5 for checkpoint selection")
    if args.epochs < 1 or args.eval_every < 1 or args.patience < 1:
        raise ValueError("epochs, eval_every, and patience must be positive")
    if args.train_batch < 1 or args.eval_batch < 1:
        raise ValueError("batch sizes must be positive")
    if args.p < 1:
        raise ValueError("P must be positive")
    if args.mc_samples < 1 or args.mc_chunk < 1:
        raise ValueError("Monte Carlo sample and chunk counts must be positive")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    dataset = InteractionDataset.from_directory(args.data_dir)
    effective_p_audit = dataset.effective_p_audit(args.p)
    graph = None if args.backbone == "mf" else dataset.normalized_graph(device)
    model = build_model(
        args.backbone,
        dataset.num_users,
        dataset.num_items,
        graph,
        dim=args.dim,
        norm=not args.no_normalize,
        lightgcn_layers=args.lightgcn_layers,
        xsimgcl_layers=args.xsimgcl_layers,
        cl_rate=args.cl_rate,
        eps=args.xsimgcl_eps,
        cl_temp=args.cl_temp,
        cl_layer=args.cl_layer,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_validation = -float("inf")
    best_epoch = 0
    best_state = None
    validations_without_improvement = 0
    history: List[Dict[str, object]] = []
    active_users = dataset.active_users
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        sequence = rng.permutation(active_users)
        if len(sequence) != len(active_users) or len(np.unique(sequence)) != len(active_users):
            raise RuntimeError("Active-user epoch permutation is not one-to-one")

        losses: List[float] = []
        effective_values: List[float] = []
        first_gradient_norm = None
        for start in range(0, len(sequence), args.train_batch):
            users_np = sequence[start : start + args.train_batch]
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = batch_loss(args, dataset, model, users_np, rng)
            if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Training loss is not a finite scalar")
            loss.backward()
            gradient_norm = finite_gradient_norm(list(model.parameters()))
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip
            )
            if not math.isfinite(float(clipped_norm.detach().cpu())):
                raise RuntimeError("Non-finite clipped gradient norm")
            if first_gradient_norm is None:
                first_gradient_norm = gradient_norm
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            effective_values.append(float(diagnostics["effective_p_mean"]))

        should_validate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_validate:
            validation = evaluate(
                model,
                dataset,
                "valid",
                cutoffs=args.cutoffs,
                batch_size=args.eval_batch,
            )
            selected_value = float(validation["ndcg@5"])
            row: Dict[str, object] = {
                "epoch": epoch,
                "mean_train_loss": float(np.mean(losses)),
                "mean_effective_p": float(np.mean(effective_values)),
                "first_batch_gradient_norm": float(first_gradient_norm),
                "validation": validation,
            }
            history.append(row)
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": row["mean_train_loss"],
                        "ndcg@5": selected_value,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            if selected_value > best_validation + 1.0e-15:
                best_validation = selected_value
                best_epoch = epoch
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if validations_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "method": "gumbel-pfy",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_ndcg@5": best_validation,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    checkpoint_hash = sha256_file(checkpoint_path)

    training_summary = {
        "status": "TRAINING_COMPLETE_NO_TEST",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": "gumbel-pfy",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ndcg@5": best_validation,
        "checkpoint_sha256": checkpoint_hash,
        "validation_history": history,
        "active_users_per_epoch": int(len(active_users)),
        "effective_p_population": effective_p_audit,
        "test_was_evaluated": False,
        "test_call_count": 0,
        "training_seconds": time.time() - started,
    }
    atomic_json(output_dir / "training_summary.json", training_summary)

    model.load_state_dict(best_state, strict=True)
    model.eval()
    test_call_count = 0

    def evaluate_test_once() -> Dict[str, float]:
        nonlocal test_call_count
        if test_call_count != 0:
            raise RuntimeError("Test evaluation was requested more than once")
        test_call_count += 1
        return evaluate(
            model,
            dataset,
            "test",
            cutoffs=args.cutoffs,
            batch_size=args.eval_batch,
        )

    test_metrics = evaluate_test_once()
    if test_call_count != 1:
        raise RuntimeError("Test evaluation must be called exactly once")
    atomic_json(
        output_dir / "test_metrics.json",
        {
            "status": "TEST_SUCCESS",
            "checkpoint_sha256": checkpoint_hash,
            "test_call_count": test_call_count,
            "test_metrics": test_metrics,
        },
    )

    final = {
        "status": "FINAL_SUCCESS",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": "gumbel-pfy",
        "seed": seed,
        "protocol": {
            "selection": "validation_ndcg@5",
            "active_user_visits_per_epoch": 1,
            "positive_sampling": "P_u=min(P,n_u), distinct train positives without replacement",
            "ranking_scope": "full catalog",
            "test_policy": "one test after the validation checkpoint is frozen",
        },
        "configuration": scientific_configuration(args),
        "effective_p_population": effective_p_audit,
        "best_epoch": best_epoch,
        "best_validation_ndcg@5": best_validation,
        "checkpoint_sha256": checkpoint_hash,
        "test_call_count": test_call_count,
        "test_metrics": test_metrics,
    }
    atomic_json(output_dir / "final_summary.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return final


def validate_completed_seed(
    payload: Dict[str, object],
    args: argparse.Namespace,
    seed: int,
) -> None:
    if payload.get("status") != "FINAL_SUCCESS":
        raise RuntimeError("Completed seed does not contain FINAL_SUCCESS")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError("Completed seed id does not match its directory")
    if payload.get("dataset") != args.dataset:
        raise RuntimeError("Completed seed uses a different dataset")
    if payload.get("backbone") != args.backbone or payload.get("method") != "gumbel-pfy":
        raise RuntimeError("Completed seed uses a different model or method")
    if payload.get("configuration") != scientific_configuration(args):
        raise RuntimeError("Completed seed uses a different configuration")
    if int(payload.get("test_call_count", 0)) != 1:
        raise RuntimeError("Completed seed does not satisfy the test-once protocol")


def aggregate_results(
    args: argparse.Namespace,
    output_dir: Path,
    seed_results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    metric_names = sorted(seed_results[0]["test_metrics"].keys())
    for result in seed_results:
        if sorted(result["test_metrics"].keys()) != metric_names:
            raise RuntimeError("Per-seed metric sets do not match")

    per_seed_rows: List[Dict[str, object]] = []
    for result in seed_results:
        row: Dict[str, object] = {
            "seed": int(result["seed"]),
            "best_epoch": int(result["best_epoch"]),
            "best_validation_ndcg@5": float(result["best_validation_ndcg@5"]),
        }
        row.update({name: float(result["test_metrics"][name]) for name in metric_names})
        per_seed_rows.append(row)

    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    aggregate_rows: List[Dict[str, object]] = []
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in per_seed_rows], dtype=np.float64)
        standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary = {"mean": float(values.mean()), "std": standard_deviation}
        aggregate_metrics[name] = summary
        aggregate_rows.append({"metric": name, **summary})

    atomic_csv(
        output_dir / "final_per_seed.csv",
        per_seed_rows,
        ["seed", "best_epoch", "best_validation_ndcg@5", *metric_names],
    )
    atomic_csv(
        output_dir / "final_aggregate.csv",
        aggregate_rows,
        ["metric", "mean", "std"],
    )

    final = {
        "status": "FINAL_SUCCESS",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": "gumbel-pfy",
        "seeds": [int(seed) for seed in args.seeds],
        "num_seeds": len(args.seeds),
        "std_definition": "sample standard deviation (ddof=1; zero for one seed)",
        "configuration": scientific_configuration(args),
        "metrics": aggregate_metrics,
    }
    atomic_json(output_dir / "final_aggregate.json", final)
    return final


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    args.seeds = normalize_integer_list(args.seeds, "seeds")
    args.cutoffs = tuple(sorted(set(normalize_integer_list(args.cutoffs, "cutoffs"))))
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must not contain duplicates")
    if min(args.cutoffs) < 1:
        raise ValueError("cutoffs must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "final_aggregate.json"
    if aggregate_path.is_file():
        existing = load_json(aggregate_path)
        if existing.get("configuration") != scientific_configuration(args):
            raise RuntimeError("Existing aggregate uses a different configuration")
        if existing.get("status") != "FINAL_SUCCESS":
            raise RuntimeError("Existing aggregate is not marked FINAL_SUCCESS")
        print("[REUSE] completed aggregate {}".format(aggregate_path), flush=True)
        return existing

    results: List[Dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = output_dir / "seed_{}".format(seed)
        final_path = seed_dir / "final_summary.json"
        if final_path.is_file():
            result = load_json(final_path)
            validate_completed_seed(result, args, seed)
            print("[REUSE] completed seed {}".format(seed), flush=True)
        else:
            if seed_dir.exists() and any(seed_dir.iterdir()):
                raise RuntimeError(
                    "Refusing to overwrite partial seed directory: {}".format(seed_dir)
                )
            result = run_seed(args, seed, seed_dir)
        results.append(result)

    final = aggregate_results(args, output_dir, results)
    print(json.dumps(final, sort_keys=True), flush=True)
    return final
