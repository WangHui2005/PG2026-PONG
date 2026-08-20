"""LODO evaluation of a classical iterative AQE+DBA baseline.

AQE updates queries from their retrieved target neighbors. DBA updates each
target only from target-target neighbors. Both operations use the same target
state within a round, so target refinement is query-independent. This isolates
PONG's query-conditioned, alternating Q->T feedback.
"""

import csv
import itertools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.spatial


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import anmrr_score, ndcg_score


FEATURE_SETS = {
    "CLIP_L14": (
        PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"
    ),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
K_VALUES = (5, 7, 10, 15)
TAU_VALUES = (0.1, 0.2)
ALPHA_VALUES = (0.3, 0.5, 0.7, 0.9)
ROUNDS = 2
GRID = [
    {"topk": topk, "tau": tau, "alpha": alpha, "n_iter": ROUNDS}
    for topk, tau, alpha in itertools.product(
        K_VALUES, TAU_VALUES, ALPHA_VALUES
    )
]


def l2_norm(values):
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def softmax_rows(values, tau):
    scaled = values / tau
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def load_setting(backbone, dataset):
    feature_dir, suffix = FEATURE_SETS[backbone]
    query = np.load(
        feature_dir / f"{dataset}_query_feats{suffix}.npy"
    ).mean(1).astype(np.float32)
    target = np.load(
        feature_dir / f"{dataset}_target_feats{suffix}.npy"
    ).mean(1).astype(np.float32)
    query_labels = np.load(
        feature_dir / f"{dataset}_query_labels{suffix}.npy"
    ).ravel().astype(int)
    target_labels = np.load(
        feature_dir / f"{dataset}_target_labels{suffix}.npy"
    ).ravel().astype(int)
    return (
        l2_norm(np.tanh(query)),
        l2_norm(np.tanh(target)),
        query_labels,
        target_labels,
    )


def topk_indices_scores(similarity, topk):
    topk = min(topk, similarity.shape[1])
    indices = np.argpartition(-similarity, topk - 1, axis=1)[:, :topk]
    scores = np.take_along_axis(similarity, indices, axis=1)
    return indices, scores


def build_fixed_dba_graph(target, topk, tau):
    """Build the classical DBA graph once from the original target bank."""
    similarity = target @ target.T
    np.fill_diagonal(similarity, -np.inf)
    indices, scores = topk_indices_scores(similarity, topk)
    return indices, softmax_rows(scores, tau)


def iterative_aqe_dba(query, target, params, dba_graph):
    """Two-round AQE+DBA with a fixed, query-independent DBA graph."""
    query = query.copy()
    target = target.copy()
    topk = params["topk"]
    tau = params["tau"]
    alpha = params["alpha"]

    target_indices, target_weights = dba_graph
    for _ in range(params["n_iter"]):
        # AQE: query receives information from retrieved targets.
        qt_similarity = query @ target.T
        query_indices, query_scores = topk_indices_scores(
            qt_similarity, topk
        )
        query_weights = softmax_rows(query_scores, tau)
        query_aggregate = np.einsum(
            "nk,nkd->nd", query_weights, target[query_indices], optimize=True
        )

        # DBA: target receives information only from a fixed target graph.
        target_aggregate = np.einsum(
            "nk,nkd->nd", target_weights, target[target_indices], optimize=True
        )

        query = l2_norm((1.0 - alpha) * query + alpha * query_aggregate)
        target = l2_norm((1.0 - alpha) * target + alpha * target_aggregate)

    return query, target


def exact_map(similarity, query_labels, target_labels):
    order = np.argsort(-similarity, axis=1)
    scores = []
    for index, ranking in enumerate(order):
        relevant = target_labels[ranking] == query_labels[index]
        positions = np.flatnonzero(relevant)
        if not len(positions):
            scores.append(0.0)
            continue
        precision = (
            np.arange(1, len(positions) + 1, dtype=np.float64)
            / (positions + 1)
        )
        scores.append(float(np.maximum.accumulate(precision[::-1])[::-1].mean()))
    return float(np.mean(scores))


def evaluate_grid_for_setting(task):
    backbone, dataset = task
    query, target, query_labels, target_labels = load_setting(backbone, dataset)
    values = []
    graphs = {
        (topk, tau): build_fixed_dba_graph(target, topk, tau)
        for topk, tau in itertools.product(K_VALUES, TAU_VALUES)
    }
    for grid_index, params in enumerate(GRID):
        refined_query, refined_target = iterative_aqe_dba(
            query, target, params, graphs[(params["topk"], params["tau"])]
        )
        values.append({
            "grid_index": grid_index,
            "mAP": exact_map(
                refined_query @ refined_target.T, query_labels, target_labels
            ),
        })
    return f"{backbone}_{dataset}", values


def full_metrics(backbone, dataset, params):
    query, target, query_labels, target_labels = load_setting(backbone, dataset)
    dba_graph = build_fixed_dba_graph(target, params["topk"], params["tau"])
    query, target = iterative_aqe_dba(query, target, params, dba_graph)
    similarity = query @ target.T
    distance = scipy.spatial.distance.cdist(query, target, "cosine")
    return {
        "mAP": exact_map(similarity, query_labels, target_labels),
        "NDCG@100": float(
            ndcg_score(distance, query_labels, target_labels, k=100)
        ),
        "ANMRR": float(anmrr_score(distance, query_labels, target_labels)),
    }


def load_pong_heldout_results():
    path = PROJECT_ROOT / "results/lodo/heldout_results.csv"
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            setting = f"{row['backbone']}_{row['heldout_dataset']}"
            rows[setting] = {
                "mAP": float(row["pong_mAP"]),
                "NDCG@100": float(row["pong_NDCG@100"]),
                "ANMRR": float(row["pong_ANMRR"]),
            }
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 PONG held-out rows, found {len(rows)}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = PROJECT_ROOT / "results/aqe_dba"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (backbone, dataset)
        for backbone in FEATURE_SETS
        for dataset in DATASETS
    ]
    grid_results = {}
    with mp.get_context("spawn").Pool(processes=4) as pool:
        for setting, values in pool.imap_unordered(
            evaluate_grid_for_setting, tasks
        ):
            grid_results[setting] = values
            print(f"Grid complete: {setting}", flush=True)

    selected_folds = []
    heldout_rows = []
    development_rows = []
    for heldout in DATASETS:
        development_settings = [
            f"{backbone}_{dataset}"
            for backbone in FEATURE_SETS
            for dataset in DATASETS
            if dataset != heldout
        ]
        mean_scores = [
            float(np.mean([
                grid_results[setting][grid_index]["mAP"]
                for setting in development_settings
            ]))
            for grid_index in range(len(GRID))
        ]
        ranking = np.argsort(-np.asarray(mean_scores))
        rank_by_index = {
            int(grid_index): rank + 1
            for rank, grid_index in enumerate(ranking)
        }
        for grid_index, score in enumerate(mean_scores):
            development_rows.append({
                "heldout_dataset": heldout,
                "grid_index": grid_index,
                "development_rank": rank_by_index[grid_index],
                "development_mean_mAP": score,
                **GRID[grid_index],
            })

        selected_index = int(np.argmax(mean_scores))
        params = GRID[selected_index]
        selected_folds.append({
            "heldout_dataset": heldout,
            "selected_grid_index": selected_index,
            "development_mean_mAP": mean_scores[selected_index],
            **params,
        })
        for backbone in FEATURE_SETS:
            heldout_rows.append({
                "heldout_dataset": heldout,
                "backbone": backbone,
                **params,
                **full_metrics(backbone, heldout, params),
            })

    pong_results = load_pong_heldout_results()
    paired_rows = []
    for row in heldout_rows:
        setting = f"{row['backbone']}_{row['heldout_dataset']}"
        pong = pong_results[setting]
        paired_rows.append({
            "setting": setting,
            "pong_mAP": pong["mAP"],
            "aqe_dba_mAP": row["mAP"],
            "delta_mAP_pp": 100 * (pong["mAP"] - row["mAP"]),
            "pong_NDCG@100": pong["NDCG@100"],
            "aqe_dba_NDCG@100": row["NDCG@100"],
            "delta_NDCG_pp": 100 * (
                pong["NDCG@100"] - row["NDCG@100"]
            ),
            "pong_ANMRR": pong["ANMRR"],
            "aqe_dba_ANMRR": row["ANMRR"],
            "delta_ANMRR_pp": 100 * (pong["ANMRR"] - row["ANMRR"]),
        })

    summary = {
        "baseline": (
            "two-round AQE+DBA: AQE updates Q from T; DBA updates T "
            "from target-target neighbors only; no query-conditioned T update"
        ),
        "selection": (
            "maximize mean mAP on three development datasets across "
            "all three backbones, then evaluate the held-out dataset"
        ),
        "grid": GRID,
        "pong_mean_mAP": float(np.mean([r["pong_mAP"] for r in paired_rows])),
        "aqe_dba_mean_mAP": float(np.mean([r["aqe_dba_mAP"] for r in paired_rows])),
        "delta_mean_mAP_pp": float(np.mean([r["delta_mAP_pp"] for r in paired_rows])),
        "pong_mean_NDCG@100": float(np.mean([r["pong_NDCG@100"] for r in paired_rows])),
        "aqe_dba_mean_NDCG@100": float(np.mean([r["aqe_dba_NDCG@100"] for r in paired_rows])),
        "delta_mean_NDCG_pp": float(np.mean([r["delta_NDCG_pp"] for r in paired_rows])),
        "pong_mean_ANMRR": float(np.mean([r["pong_ANMRR"] for r in paired_rows])),
        "aqe_dba_mean_ANMRR": float(np.mean([r["aqe_dba_ANMRR"] for r in paired_rows])),
        "delta_mean_ANMRR_pp": float(np.mean([r["delta_ANMRR_pp"] for r in paired_rows])),
        "pong_mAP_wins": sum(r["delta_mAP_pp"] > 0 for r in paired_rows),
        "pong_NDCG_wins": sum(r["delta_NDCG_pp"] > 0 for r in paired_rows),
        "pong_ANMRR_wins": sum(r["delta_ANMRR_pp"] < 0 for r in paired_rows),
    }
    write_csv(output_dir / "selected_parameters.csv", selected_folds)
    write_csv(output_dir / "heldout_results.csv", heldout_rows)
    write_csv(output_dir / "development_grid_scores.csv", development_rows)
    write_csv(output_dir / "paired_vs_pong.csv", paired_rows)
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
