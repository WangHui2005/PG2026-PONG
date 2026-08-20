"""LODO-parameterized robustness evaluations for PONG.

The script separates two questions: (i) partial-view observations, evaluated
by uniformly subsampling the 24 precomputed rendered views, and (ii) error
propagation in the low initial-neighborhood-purity subset.  It deliberately
does not claim robustness to raw mesh noise or real scans, which are absent
from the available benchmarks.
"""

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.pong import l2_normalize, refine

OUT = ROOT / "results/robustness"
FEATURE_SETS = {
    "CLIP_L14": (ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (ROOT / "data/openclip_L14", ""),
    "DINOv2": (ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
VIEW_COUNTS = (24, 12, 6)
NOISE_SIGMAS = (0.02, 0.05)
NOISE_SEEDS = (11, 29, 47)


def softmax(values, tau):
    values = values / tau
    values -= values.max(axis=1, keepdims=True)
    weights = np.exp(values)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def topk(similarity, size):
    size = min(size, similarity.shape[1])
    indices = np.argpartition(-similarity, size - 1, axis=1)[:, :size]
    return indices, np.take_along_axis(similarity, indices, axis=1)


def exact_map(similarity, query_labels, target_labels):
    order = np.argsort(-similarity, axis=1)
    values = []
    for query_index, ranking in enumerate(order):
        relevant = target_labels[ranking] == query_labels[query_index]
        positions = np.flatnonzero(relevant)
        if not len(positions):
            values.append(0.0)
        else:
            precision = np.arange(1, len(positions) + 1) / (positions + 1)
            values.append(float(np.maximum.accumulate(precision[::-1])[::-1].mean()))
    return float(np.mean(values))


def pong(query, target, params):
    return refine(
        query,
        target,
        k=params["topk"],
        tau=params["tau"],
        lambda_min=params["lam_min"],
        lambda_max=params["lam_max"],
        rounds=params["n_iter"],
    )


def selected_params():
    rows = csv.DictReader((ROOT / "results/lodo/selected_parameters.csv").open())
    return {row["heldout_dataset"]: {
        "topk": int(row["topk"]), "tau": float(row["tau"]),
        "lam_min": float(row["lam_min"]), "lam_max": float(row["lam_max"]),
        "n_iter": int(row["n_iter"]),
    } for row in rows}


def load(backbone, dataset, view_count):
    directory, suffix = FEATURE_SETS[backbone]
    query_all = np.load(directory / f"{dataset}_query_feats{suffix}.npy").astype(np.float32)
    target_all = np.load(directory / f"{dataset}_target_feats{suffix}.npy").astype(np.float32)
    if query_all.shape[1] != 24 or target_all.shape[1] != 24:
        raise RuntimeError("Expected 24-view features")
    view_ids = np.linspace(0, 23, view_count, dtype=int)
    query = l2_normalize(np.tanh(query_all[:, view_ids].mean(1)))
    target = l2_normalize(np.tanh(target_all[:, view_ids].mean(1)))
    query_labels = np.load(directory / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(directory / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return query, target, query_labels, target_labels, view_ids.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params_by_dataset = selected_params()
    view_rows, purity_rows, noise_rows = [], [], []
    for backbone in FEATURE_SETS:
        for dataset in DATASETS:
            params = params_by_dataset[dataset]
            for view_count in VIEW_COUNTS:
                query, target, ql, tl, ids = load(backbone, dataset, view_count)
                baseline = exact_map(query @ target.T, ql, tl)
                refined_q, refined_t = pong(query, target, params)
                refined = exact_map(refined_q @ refined_t.T, ql, tl)
                view_rows.append({
                    "setting": f"{backbone}_{dataset}", "views": view_count,
                    "view_indices": " ".join(map(str, ids)), "baseline_mAP": baseline,
                    "pong_mAP": refined, "delta_mAP_pp": 100 * (refined - baseline),
                })

            # Initial purity and PONG gain both evaluated with the complete 24-view input.
            query, target, ql, tl, _ = load(backbone, dataset, 24)
            indices, _ = topk(query @ target.T, params["topk"])
            purity = (tl[indices] == ql[:, None]).mean(1)
            subset = purity < 0.4
            refined_q, refined_t = pong(query, target, params)
            baseline_subset = exact_map((query @ target.T)[subset], ql[subset], tl)
            refined_subset = exact_map((refined_q @ refined_t.T)[subset], ql[subset], tl)
            purity_rows.append({
                "setting": f"{backbone}_{dataset}", "n_queries": len(query),
                "n_low_purity_queries": int(subset.sum()), "purity_threshold": "<0.4",
                "baseline_low_purity_mAP": baseline_subset,
                "pong_low_purity_mAP": refined_subset,
                "delta_low_purity_mAP_pp": 100 * (refined_subset - baseline_subset),
            })

            # Descriptor-space perturbation is an explicit controlled proxy for
            # representation noise, not a claim about raw geometric corruption.
            clean_query, clean_target, ql, tl, _ = load(backbone, dataset, 24)
            for sigma in NOISE_SIGMAS:
                for seed in NOISE_SEEDS:
                    generator = np.random.default_rng(seed)
                    noisy_query = l2_normalize(clean_query + generator.normal(0, sigma, clean_query.shape).astype(np.float32))
                    noisy_target = l2_normalize(clean_target + generator.normal(0, sigma, clean_target.shape).astype(np.float32))
                    baseline = exact_map(noisy_query @ noisy_target.T, ql, tl)
                    refined_q, refined_t = pong(noisy_query, noisy_target, params)
                    refined = exact_map(refined_q @ refined_t.T, ql, tl)
                    noise_rows.append({
                        "setting": f"{backbone}_{dataset}", "sigma": sigma, "seed": seed,
                        "baseline_mAP": baseline, "pong_mAP": refined,
                        "delta_mAP_pp": 100 * (refined - baseline),
                    })
            print(backbone, dataset, "complete", flush=True)

    for name, rows in (("partial_views.csv", view_rows), ("low_purity.csv", purity_rows), ("descriptor_noise.csv", noise_rows)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    summary = {}
    for views in VIEW_COUNTS:
        rows = [row for row in view_rows if row["views"] == views]
        deltas = [row["delta_mAP_pp"] for row in rows]
        summary[f"views_{views}"] = {
            "mean_baseline_mAP": float(np.mean([r["baseline_mAP"] for r in rows])),
            "mean_pong_mAP": float(np.mean([r["pong_mAP"] for r in rows])),
            "mean_delta_mAP_pp": float(np.mean(deltas)),
            "positive_settings": int(sum(value > 0 for value in deltas)),
        }
    deltas = [row["delta_low_purity_mAP_pp"] for row in purity_rows]
    summary["low_purity"] = {
        "mean_delta_mAP_pp": float(np.mean(deltas)),
        "positive_settings": int(sum(value > 0 for value in deltas)),
        "negative_settings": int(sum(value < 0 for value in deltas)),
        "near_zero_settings": int(sum(abs(value) < 0.01 for value in deltas)),
    }
    for sigma in NOISE_SIGMAS:
        rows = [row for row in noise_rows if row["sigma"] == sigma]
        deltas = [row["delta_mAP_pp"] for row in rows]
        setting_deltas = [
            np.mean([row["delta_mAP_pp"] for row in rows if row["setting"] == setting])
            for setting in sorted({row["setting"] for row in rows})
        ]
        summary[f"descriptor_noise_sigma_{sigma}"] = {
            "mean_baseline_mAP": float(np.mean([r["baseline_mAP"] for r in rows])),
            "mean_pong_mAP": float(np.mean([r["pong_mAP"] for r in rows])),
            "mean_delta_mAP_pp": float(np.mean(deltas)),
            "positive_settings_mean_over_seeds": int(sum(value > 0 for value in setting_deltas)),
            "seeds": list(NOISE_SEEDS),
        }
    with (OUT / "metadata.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
