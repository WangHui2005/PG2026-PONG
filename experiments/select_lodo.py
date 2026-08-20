"""Leave-one-dataset-out hyperparameter selection for PONG.

For each held-out dataset, select one shared configuration by average mAP on
the other three datasets across all three backbones. The held-out labels and
metrics are never used for selection. Evaluation always uses the features after
exactly two rounds; there is no label-based best-iteration selection.
"""

import argparse
import csv
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

from methods.config import lodo_grid
from methods.pong import l2_normalize, refine
from utils.metrics import anmrr_score, ndcg_score


FEATURE_SETS = {
    "CLIP_L14": (PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
GRID = lodo_grid()

TEDA = {
    "CLIP_L14": {
        "esb": (0.6092, 0.2513, 0.4399), "ntu": (0.6163, 0.2744, 0.4177),
        "abo": (0.6559, 0.5780, 0.3702), "mn40": (0.6394, 0.7383, 0.3827),
    },
    "OpenCLIP_L14": {
        "esb": (0.6505, 0.2606, 0.4024), "ntu": (0.6647, 0.2891, 0.3774),
        "abo": (0.6997, 0.5866, 0.3236), "mn40": (0.7342, 0.7965, 0.2994),
    },
    "DINOv2": {
        "esb": (0.6176, 0.2522, 0.4342), "ntu": (0.6291, 0.2769, 0.4066),
        "abo": (0.6783, 0.5914, 0.3558), "mn40": (0.7012, 0.7725, 0.3342),
    },
}


def load_setting(backbone, dataset):
    feature_dir, suffix = FEATURE_SETS[backbone]
    query = np.load(feature_dir / f"{dataset}_query_feats{suffix}.npy").mean(1).astype(np.float32)
    target = np.load(feature_dir / f"{dataset}_target_feats{suffix}.npy").mean(1).astype(np.float32)
    query_labels = np.load(feature_dir / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(feature_dir / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return l2_normalize(np.tanh(query)), l2_normalize(np.tanh(target)), query_labels, target_labels


def pong_final(query, target, params):
    return refine(
        query,
        target,
        k=params["topk"],
        tau=params["tau"],
        lambda_min=params["lam_min"],
        lambda_max=params["lam_max"],
        rounds=params["n_iter"],
    )


def exact_map(similarity, query_labels, target_labels):
    order = np.argsort(-similarity, axis=1)
    scores = []
    for index, ranking in enumerate(order):
        relevant = target_labels[ranking] == query_labels[index]
        positions = np.flatnonzero(relevant)
        if not len(positions):
            scores.append(0.0)
            continue
        precision = np.arange(1, len(positions) + 1, dtype=np.float64) / (positions + 1)
        interpolated = np.maximum.accumulate(precision[::-1])[::-1]
        scores.append(float(interpolated.mean()))
    return float(np.mean(scores))


def evaluate_grid_for_setting(task):
    backbone, dataset = task
    query, target, query_labels, target_labels = load_setting(backbone, dataset)
    values = []
    for grid_index, params in enumerate(GRID):
        refined_query, refined_target = pong_final(query, target, params)
        score = exact_map(refined_query @ refined_target.T, query_labels, target_labels)
        values.append({"grid_index": grid_index, "mAP": score})
    return f"{backbone}_{dataset}", values


def full_metrics(backbone, dataset, params):
    query, target, query_labels, target_labels = load_setting(backbone, dataset)
    query, target = pong_final(query, target, params)
    distance = scipy.spatial.distance.cdist(query, target, "cosine")
    map_value = exact_map(query @ target.T, query_labels, target_labels)
    return {
        "mAP": map_value,
        "NDCG@100": float(ndcg_score(distance, query_labels, target_labels, k=100)),
        "ANMRR": float(anmrr_score(distance, query_labels, target_labels)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_grid_results(tasks):
    """Evaluate only the settings permitted for the current LODO fold."""
    grid_results = {}
    with mp.get_context("spawn").Pool(processes=min(9, len(tasks))) as pool:
        for setting, values in pool.imap_unordered(evaluate_grid_for_setting, tasks):
            grid_results[setting] = values
            print(f"Development grid complete: {setting}", flush=True)
    return grid_results


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_folds = []
    heldout_rows = []
    development_grid_rows = []
    development_setting_rows = []
    for heldout in DATASETS:
        # This order is deliberate: the held-out dataset is never evaluated on
        # candidate configurations before this fold's parameter is selected.
        dev_tasks = [
            (backbone, dataset)
            for backbone in FEATURE_SETS
            for dataset in DATASETS
            if dataset != heldout
        ]
        grid_results = collect_grid_results(dev_tasks)
        dev_settings = [
            f"{backbone}_{dataset}"
            for backbone, dataset in dev_tasks
        ]
        mean_scores = []
        for grid_index in range(len(GRID)):
            mean_scores.append(np.mean([
                grid_results[setting][grid_index]["mAP"] for setting in dev_settings
            ]))
        ranking = np.argsort(-np.asarray(mean_scores))
        rank_by_index = {
            int(grid_index): rank + 1 for rank, grid_index in enumerate(ranking)
        }
        for grid_index, mean_score in enumerate(mean_scores):
            development_grid_rows.append({
                "heldout_dataset": heldout,
                "grid_index": grid_index,
                "development_rank": rank_by_index[grid_index],
                "development_mean_mAP": float(mean_score),
                **GRID[grid_index],
            })

        for setting in sorted(grid_results):
            backbone, dataset = setting.rsplit("_", 1)
            for value in grid_results[setting]:
                development_setting_rows.append({
                    "heldout_dataset": heldout,
                    "setting": setting,
                    "backbone": backbone,
                    "dataset": dataset,
                    **value,
                    **GRID[value["grid_index"]],
                })
        selected_index = int(np.argmax(mean_scores))
        selected_params = GRID[selected_index]
        selected_folds.append({
            "heldout_dataset": heldout,
            "selected_grid_index": selected_index,
            "development_mean_mAP": float(mean_scores[selected_index]),
            **selected_params,
        })

        for backbone in FEATURE_SETS:
            measured = full_metrics(backbone, heldout, selected_params)
            teda_map, teda_ndcg, teda_anmrr = TEDA[backbone][heldout]
            heldout_rows.append({
                "heldout_dataset": heldout,
                "backbone": backbone,
                **selected_params,
                "teda_mAP": teda_map,
                "pong_mAP": measured["mAP"],
                "delta_mAP_pp": 100 * (measured["mAP"] - teda_map),
                "teda_NDCG@100": teda_ndcg,
                "pong_NDCG@100": measured["NDCG@100"],
                "delta_NDCG_pp": 100 * (measured["NDCG@100"] - teda_ndcg),
                "teda_ANMRR": teda_anmrr,
                "pong_ANMRR": measured["ANMRR"],
                "delta_ANMRR_pp": 100 * (measured["ANMRR"] - teda_anmrr),
            })

    write_csv(output_dir / "selected_parameters.csv", selected_folds)
    write_csv(output_dir / "heldout_results.csv", heldout_rows)
    write_csv(output_dir / "development_grid_scores.csv", development_grid_rows)
    write_csv(output_dir / "development_setting_grid.csv", development_setting_rows)
    with (output_dir / "lodo_results.json").open("w") as handle:
        json.dump({
            "protocol": {
                "selection": "maximize mean mAP on 3 development datasets x 3 backbones",
                "evaluation": "apply selected parameters unchanged to the held-out dataset x 3 backbones",
                "round_policy": "exactly 2 rounds; no label-based best-iteration selection",
                "execution_order": "for each fold, evaluate development grids, select once, then evaluate the held-out dataset",
                "grid": GRID,
            },
            "selected_folds": selected_folds,
            "heldout_results": heldout_rows,
        }, handle, indent=2)

    map_wins = sum(row["delta_mAP_pp"] > 0 for row in heldout_rows)
    ndcg_wins = sum(row["delta_NDCG_pp"] > 0 for row in heldout_rows)
    anmrr_wins = sum(row["delta_ANMRR_pp"] < 0 for row in heldout_rows)
    print(f"LODO summary: mAP {map_wins}/12, NDCG {ndcg_wins}/12, ANMRR {anmrr_wins}/12", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/lodo",
        help="Directory for LODO selection and held-out evaluation outputs.",
    )
    args = parser.parse_args()
    main(args.output_dir)
