"""Evaluate PONG's transductive query-buffer behavior.

The protocol uses each held-out dataset's fold-selected LODO configuration:
  - Three folds use (5, 0.2, 0.0, 0.7, 2).
  - The MN40 fold uses (5, 0.2, 0.1, 0.5, 2).
  - No label-based iteration selection.
  - Reset experiments start every query batch from the original targets.
  - Reverse top-k is min(k, current query-batch size).
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.config import dominant_lodo_params
from methods.pong import l2_normalize, refine
from utils.metrics import anmrr_score, map_score, ndcg_score


FEATURE_SETS = {
    "CLIP_L14": (PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
OOD_SOURCE = {"esb": "ntu", "ntu": "esb", "abo": "mn40", "mn40": "abo"}
DEFAULT_PARAMS = dominant_lodo_params()
RANDOM_SEEDS = tuple(range(5))


def load_fold_params():
    path = PROJECT_ROOT / "results/lodo/selected_parameters.csv"
    selected = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            selected[row["heldout_dataset"]] = {
                "topk": int(row["topk"]),
                "tau": float(row["tau"]),
                "lam_min": float(row["lam_min"]),
                "lam_max": float(row["lam_max"]),
                "n_iter": int(row["n_iter"]),
            }
    if set(selected) != set(DATASETS):
        raise RuntimeError(f"Expected four LODO folds, found {sorted(selected)}")
    return selected


PARAMS_BY_DATASET = load_fold_params()


def softmax_rows(values, tau):
    scaled = values / tau
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def load_setting(backbone, dataset):
    feature_dir, suffix = FEATURE_SETS[backbone]
    query = np.load(feature_dir / f"{dataset}_query_feats{suffix}.npy").mean(1).astype(np.float32)
    target = np.load(feature_dir / f"{dataset}_target_feats{suffix}.npy").mean(1).astype(np.float32)
    query_labels = np.load(feature_dir / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(feature_dir / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return l2_normalize(np.tanh(query)), l2_normalize(np.tanh(target)), query_labels, target_labels


def pong_refine(query, target, params=None):
    params = DEFAULT_PARAMS if params is None else params
    return refine(
        query,
        target,
        k=params["topk"],
        tau=params["tau"],
        lambda_min=params["lam_min"],
        lambda_max=params["lam_max"],
        rounds=params["n_iter"],
    )


def target_self_refine(target, params):
    target = target.copy()
    topk = min(params["topk"], len(target) - 1)

    for _ in range(params["n_iter"]):
        similarity = target @ target.T
        np.fill_diagonal(similarity, -np.inf)
        indices = np.argpartition(-similarity, topk - 1, axis=1)[:, :topk]
        scores = np.take_along_axis(similarity, indices, axis=1)
        weights = softmax_rows(scores, params["tau"])
        aggregate = np.einsum("nk,nkd->nd", weights, target[indices], optimize=True)
        confidence = np.clip(scores.mean(axis=1), 0, 1)
        lambdas = (
            params["lam_min"]
            + (params["lam_max"] - params["lam_min"]) * confidence
        )[:, None]
        target = l2_normalize((1 - lambdas) * target + lambdas * aggregate)

    return target


def online_query_scores(query, fixed_target, params):
    rows = np.empty((len(query), len(fixed_target)), dtype=np.float32)
    start = time.perf_counter()
    for index in range(len(query)):
        refined = query[index : index + 1].copy()
        for _ in range(params["n_iter"]):
            similarity = refined @ fixed_target.T
            k_eff = min(params["topk"], len(fixed_target))
            neighbors = np.argpartition(-similarity, k_eff - 1, axis=1)[:, :k_eff]
            scores = np.take_along_axis(similarity, neighbors, axis=1)
            weights = softmax_rows(scores, params["tau"])
            aggregate = np.einsum(
                "nk,nkd->nd", weights, fixed_target[neighbors], optimize=True
            )
            confidence = np.clip(scores.mean(axis=1), 0, 1)
            lambdas = (
                params["lam_min"]
                + (params["lam_max"] - params["lam_min"]) * confidence
            )[:, None]
            refined = l2_normalize((1 - lambdas) * refined + lambdas * aggregate)
        rows[index] = refined @ fixed_target.T
    elapsed = time.perf_counter() - start
    return rows, 1000 * elapsed / len(query)


def metric_values(similarity, query_labels, target_labels):
    distance = 1.0 - similarity
    return {
        "mAP": float(map_score(distance, query_labels, target_labels)),
        "NDCG@100": float(ndcg_score(distance, query_labels, target_labels, k=100)),
        "ANMRR": float(anmrr_score(distance, query_labels, target_labels)),
    }


def random_partition(length, batch_size, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(length)
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


def reset_partition_scores(query, target, batches, params):
    similarity = np.empty((len(query), len(target)), dtype=np.float32)
    for indices in batches:
        refined_q, refined_t = pong_refine(query[indices], target, params)
        similarity[indices] = refined_q @ refined_t.T
    return similarity


def persistent_partition_scores(query, target, batches, params):
    similarity = np.empty((len(query), len(target)), dtype=np.float32)
    current_target = target.copy()
    for indices in batches:
        refined_q, current_target = pong_refine(query[indices], current_target, params)
        similarity[indices] = refined_q @ current_target.T
    drift = float(np.mean(1.0 - np.sum(current_target * target, axis=1)))
    return similarity, drift


def topk_overlap(reference, measured, k=100):
    k = min(k, reference.shape[1])
    ref_top = np.argpartition(-reference, k - 1, axis=1)[:, :k]
    measured_top = np.argpartition(-measured, k - 1, axis=1)[:, :k]
    overlaps = [
        len(np.intersect1d(ref_top[index], measured_top[index])) / k
        for index in range(len(reference))
    ]
    return float(np.mean(overlaps))


def distinct_class_indices(labels, count, rng):
    classes = rng.choice(np.unique(labels), size=count, replace=False)
    return np.asarray([rng.choice(np.flatnonzero(labels == cls)) for cls in classes])


def composition_experiment(backbone, dataset, query, target, query_labels, target_labels, params):
    ood_query, _, _, _ = load_setting(backbone, OOD_SOURCE[dataset])
    condition_scores = {name: [] for name in ("clean", "class_concentrated", "duplicate", "ood")}
    anchor_labels = []
    overlap_values = {name: [] for name in ("class_concentrated", "duplicate", "ood")}

    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(1000 + seed)
        anchors = distinct_class_indices(query_labels, 5, rng)
        remaining = np.setdiff1d(np.arange(len(query)), anchors)
        clean_context = rng.choice(remaining, size=5, replace=False)

        anchor_classes = set(query_labels[anchors].tolist())
        candidate_classes = [
            cls for cls in np.unique(query_labels) if cls not in anchor_classes
        ]
        concentrated_class = rng.choice(candidate_classes)
        concentrated_context = np.flatnonzero(query_labels == concentrated_class)
        concentrated_context = rng.choice(concentrated_context, size=5, replace=False)

        duplicate_context = clean_context.copy()
        duplicate_context[:3] = clean_context[3]
        ood_indices = rng.choice(len(ood_query), size=3, replace=False)

        batches = {
            "clean": np.concatenate([query[anchors], query[clean_context]], axis=0),
            "class_concentrated": np.concatenate(
                [query[anchors], query[concentrated_context]], axis=0
            ),
            "duplicate": np.concatenate(
                [query[anchors], query[duplicate_context]], axis=0
            ),
            "ood": np.concatenate(
                [query[anchors], query[clean_context[:2]], ood_query[ood_indices]], axis=0
            ),
        }
        seed_scores = {}
        for name, batch_query in batches.items():
            refined_q, refined_t = pong_refine(batch_query, target, params)
            seed_scores[name] = refined_q[:5] @ refined_t.T
            condition_scores[name].append(seed_scores[name])
        anchor_labels.extend(query_labels[anchors].tolist())
        for name in overlap_values:
            overlap_values[name].append(
                topk_overlap(seed_scores["clean"], seed_scores[name])
            )

    rows = []
    labels = np.asarray(anchor_labels)
    clean_metrics = metric_values(
        np.concatenate(condition_scores["clean"], axis=0), labels, target_labels
    )
    for name, values in condition_scores.items():
        measured = metric_values(np.concatenate(values, axis=0), labels, target_labels)
        rows.append({
            "condition": name,
            **measured,
            "delta_mAP_vs_clean_pp": 100 * (measured["mAP"] - clean_metrics["mAP"]),
            "top100_overlap_vs_clean": (
                1.0 if name == "clean" else float(np.mean(overlap_values[name]))
            ),
        })
    return rows


def evaluate_setting(task):
    backbone, dataset = task
    params = PARAMS_BY_DATASET[dataset]
    setting = f"{backbone}_{dataset}"
    query, target, query_labels, target_labels = load_setting(backbone, dataset)

    batch_rows = []
    full_scores = None
    for batch_label, batch_size, seeds in (
        ("single", 1, (0,)),
        ("5", 5, RANDOM_SEEDS),
        ("10", 10, RANDOM_SEEDS),
        ("full", len(query), (0,)),
    ):
        score_runs = []
        for seed in seeds:
            batches = random_partition(len(query), batch_size, seed)
            scores = reset_partition_scores(query, target, batches, params)
            score_runs.append((seed, len(batches), scores))
            if batch_label == "full":
                full_scores = scores
        reference_scores = score_runs[0][2]
        for seed, num_batches, scores in score_runs:
            batch_rows.append({
                "setting": setting,
                "backbone": backbone,
                "dataset": dataset,
                "batch_size": batch_label,
                "seed": seed,
                "num_batches": num_batches,
                "top100_overlap_vs_seed0": topk_overlap(reference_scores, scores),
                **metric_values(scores, query_labels, target_labels),
            })

    fixed_batches = random_partition(len(query), 10, 2026)
    reset_scores = reset_partition_scores(query, target, fixed_batches, params)
    stream_rows = [{
        "setting": setting,
        "backbone": backbone,
        "dataset": dataset,
        "policy": "reset",
        "order_seed": 0,
        "retained_target_drift": 0.0,
        **metric_values(reset_scores, query_labels, target_labels),
    }]
    order_rng = np.random.default_rng(3000)
    for order_seed in RANDOM_SEEDS:
        order = order_rng.permutation(len(fixed_batches))
        ordered_batches = [fixed_batches[index] for index in order]
        scores, drift = persistent_partition_scores(query, target, ordered_batches, params)
        stream_rows.append({
            "setting": setting,
            "backbone": backbone,
            "dataset": dataset,
            "policy": "persistent",
            "order_seed": order_seed,
            "retained_target_drift": drift,
            **metric_values(scores, query_labels, target_labels),
        })

    composition_rows = composition_experiment(
        backbone, dataset, query, target, query_labels, target_labels, params
    )
    for row in composition_rows:
        row.update({"setting": setting, "backbone": backbone, "dataset": dataset})

    baseline_scores = query @ target.T
    offline_start = time.perf_counter()
    fixed_target = target_self_refine(target, params)
    offline_seconds = time.perf_counter() - offline_start
    online_scores, online_ms = online_query_scores(query, fixed_target, params)
    online_rows = []
    for method, scores, offline_time, query_time in (
        ("no_refinement", baseline_scores, 0.0, 0.0),
        ("pong_full_batch", full_scores, 0.0, 0.0),
        ("target_precomputed_single", online_scores, offline_seconds, online_ms),
    ):
        online_rows.append({
            "setting": setting,
            "backbone": backbone,
            "dataset": dataset,
            "method": method,
            "offline_target_seconds": offline_time,
            "online_ms_per_query": query_time,
            **metric_values(scores, query_labels, target_labels),
        })

    return {
        "setting": setting,
        "batch": batch_rows,
        "stream": stream_rows,
        "composition": composition_rows,
        "online": online_rows,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/query_buffer",
    )
    args = parser.parse_args()

    all_tasks = [(backbone, dataset) for backbone in FEATURE_SETS for dataset in DATASETS]
    if args.setting:
        requested = set(args.setting)
        all_tasks = [
            task for task in all_tasks if f"{task[0]}_{task[1]}" in requested
        ]
        missing = requested - {f"{task[0]}_{task[1]}" for task in all_tasks}
        if missing:
            raise ValueError(f"Unknown settings: {sorted(missing)}")

    results = []
    with mp.get_context("spawn").Pool(processes=min(args.workers, len(all_tasks))) as pool:
        for result in pool.imap_unordered(evaluate_setting, all_tasks):
            results.append(result)
            print(f"Complete: {result['setting']}", flush=True)

    results.sort(key=lambda item: item["setting"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("batch", "batch_size.csv"),
        ("stream", "stream_policy.csv"),
        ("composition", "composition_stress.csv"),
        ("online", "online_variant.csv"),
    ):
        rows = [row for result in results for row in result[key]]
        write_csv(output_dir / filename, rows)

    with (output_dir / "protocol.json").open("w") as handle:
        json.dump({
            "parameters_by_heldout_dataset": PARAMS_BY_DATASET,
            "random_seeds": RANDOM_SEEDS,
            "batch_sizes": ["single", 5, 10, "full"],
            "target_policy": "reset for formal PONG; persistent only as a diagnostic",
            "iteration_policy": "exactly two rounds; no label-based selection",
            "reverse_topk": "min(k, query_batch_size)",
            "settings": [result["setting"] for result in results],
        }, handle, indent=2)
    print(f"Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
