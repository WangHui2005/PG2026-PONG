"""Post-hoc ablations evaluated with the fold-selected LODO configurations.

This script does not select parameters. For each held-out dataset, it loads the
configuration already selected from the other three datasets and evaluates
direction/update variants and round trajectories on the held-out settings.
Labels are used only for reporting metrics after each variant has finished.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.pong import _topk, _weights, l2_normalize, refine


FEATURE_SETS = {
    "CLIP_L14": (PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
VARIANTS = (
    "baseline",
    "query_only",
    "target_only",
    "simultaneous",
    "alternating_fixed",
    "pong_adaptive",
)


def load_setting(backbone: str, dataset: str):
    feature_dir, suffix = FEATURE_SETS[backbone]
    query = np.load(feature_dir / f"{dataset}_query_feats{suffix}.npy").mean(1).astype(np.float32)
    target = np.load(feature_dir / f"{dataset}_target_feats{suffix}.npy").mean(1).astype(np.float32)
    query_labels = np.load(feature_dir / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(feature_dir / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return l2_normalize(np.tanh(query)), l2_normalize(np.tanh(target)), query_labels, target_labels


def load_fold_params() -> dict[str, dict[str, float | int]]:
    path = PROJECT_ROOT / "results/lodo/selected_parameters.csv"
    selected = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            selected[row["heldout_dataset"]] = {
                "k": int(row["topk"]),
                "tau": float(row["tau"]),
                "lambda_min": float(row["lam_min"]),
                "lambda_max": float(row["lam_max"]),
                "rounds": int(row["n_iter"]),
            }
    if set(selected) != set(DATASETS):
        raise RuntimeError(f"Expected four LODO folds, found {sorted(selected)}")
    return selected


def exact_map(similarity: np.ndarray, query_labels: np.ndarray, target_labels: np.ndarray) -> float:
    order = np.argsort(-similarity, axis=1)
    scores = []
    for index, ranking in enumerate(order):
        relevant = target_labels[ranking] == query_labels[index]
        positions = np.flatnonzero(relevant)
        if not len(positions):
            scores.append(0.0)
            continue
        precision = np.arange(1, len(positions) + 1, dtype=np.float64) / (positions + 1)
        scores.append(float(np.maximum.accumulate(precision[::-1])[::-1].mean()))
    return float(np.mean(scores))


def update(receivers, donors, *, k, tau, lambda_min, lambda_max, fixed_lambda=None):
    indices, scores = _topk(receivers @ donors.T, k)
    aggregate = np.einsum("nk,nkd->nd", _weights(scores, tau), donors[indices], optimize=True)
    if fixed_lambda is None:
        confidence = np.clip(scores.mean(axis=1), 0.0, 1.0)
        blend = lambda_min + (lambda_max - lambda_min) * confidence
    else:
        blend = np.full(len(receivers), fixed_lambda, dtype=np.float32)
    return l2_normalize((1.0 - blend[:, None]) * receivers + blend[:, None] * aggregate)


def run_variant(query, target, params, variant):
    query = query.copy()
    target = target.copy()
    if variant == "baseline":
        return query, target
    if variant == "pong_adaptive":
        return refine(query, target, **params)

    fixed_lambda = None
    if variant == "alternating_fixed":
        fixed_lambda = 0.5 * (params["lambda_min"] + params["lambda_max"])

    for _ in range(params["rounds"]):
        if variant == "query_only":
            query = update(query, target, **{k: params[k] for k in ("k", "tau", "lambda_min", "lambda_max")})
        elif variant == "target_only":
            target = update(target, query, **{k: params[k] for k in ("k", "tau", "lambda_min", "lambda_max")})
        elif variant == "simultaneous":
            old_query, old_target = query, target
            query = update(old_query, old_target, **{k: params[k] for k in ("k", "tau", "lambda_min", "lambda_max")})
            target = update(old_target, old_query, **{k: params[k] for k in ("k", "tau", "lambda_min", "lambda_max")})
        elif variant == "alternating_fixed":
            common = {k: params[k] for k in ("k", "tau", "lambda_min", "lambda_max")}
            query = update(query, target, fixed_lambda=fixed_lambda, **common)
            target = update(target, query, fixed_lambda=fixed_lambda, **common)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return query, target


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rounds(rows: list[dict], output_path: Path):
    labels = {"esb": "OS-ESB-core", "ntu": "OS-NTU-core", "abo": "OS-ABO-core", "mn40": "OS-MN40-core"}
    colors = {"CLIP_L14": "#4477AA", "OpenCLIP_L14": "#66CCEE", "DINOv2": "#CCBB44"}
    markers = {"CLIP_L14": "o", "OpenCLIP_L14": "s", "DINOv2": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for ax, dataset in zip(axes.ravel(), DATASETS):
        for backbone in FEATURE_SETS:
            values = [row for row in rows if row["heldout_dataset"] == dataset and row["backbone"] == backbone]
            values.sort(key=lambda row: int(row["rounds"]))
            ax.plot(
                [int(row["rounds"]) for row in values],
                [100.0 * float(row["mAP"]) for row in values],
                marker=markers[backbone],
                markersize=3.0,
                linewidth=0.9,
                color=colors[backbone],
                label=backbone.replace("_L14", "") if dataset == "esb" else None,
            )
            peak_index = int(np.argmax([float(row["mAP"]) for row in values]))
            peak = values[peak_index]
            ax.plot(
                int(peak["rounds"]),
                100.0 * float(peak["mAP"]),
                marker="D",
                markersize=4.8,
                markerfacecolor=colors[backbone],
                markeredgecolor="white",
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
            )
        ax.axvline(2, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(labels[dataset], fontsize=8)
        ax.set_xlabel("Propagation rounds", fontsize=7)
        ax.set_ylabel("mAP (%)", fontsize=7)
        ax.set_xticks(range(0, 6))
        ax.tick_params(labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    output_dir = PROJECT_ROOT / "results/ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_params = load_fold_params()
    directional_rows = []
    rounds_rows = []

    for dataset in DATASETS:
        params = fold_params[dataset]
        for backbone in FEATURE_SETS:
            query, target, query_labels, target_labels = load_setting(backbone, dataset)
            for variant in VARIANTS:
                refined_query, refined_target = run_variant(query, target, params, variant)
                directional_rows.append({
                    "heldout_dataset": dataset,
                    "backbone": backbone,
                    "variant": variant,
                    "mAP": exact_map(refined_query @ refined_target.T, query_labels, target_labels),
                    **params,
                })
            for rounds in range(0, 6):
                if rounds == 0:
                    refined_query, refined_target = query, target
                else:
                    refined_query, refined_target = refine(query, target, **{**params, "rounds": rounds})
                rounds_rows.append({
                    "heldout_dataset": dataset,
                    "backbone": backbone,
                    "rounds": rounds,
                    "mAP": exact_map(refined_query @ refined_target.T, query_labels, target_labels),
                    "selected_rounds": params["rounds"],
                    "k": params["k"],
                    "tau": params["tau"],
                    "lambda_min": params["lambda_min"],
                    "lambda_max": params["lambda_max"],
                })
            print(f"Complete: {backbone} / {dataset}", flush=True)

    directional_summary = []
    for variant in VARIANTS:
        values = [row["mAP"] for row in directional_rows if row["variant"] == variant]
        directional_summary.append({
            "variant": variant,
            "mean_mAP": float(np.mean(values)),
            "delta_over_baseline_pp": 100.0 * (float(np.mean(values)) - float(np.mean([
                row["mAP"] for row in directional_rows if row["variant"] == "baseline"
            ]))),
        })

    write_csv(output_dir / "directional_per_setting.csv", directional_rows)
    write_csv(output_dir / "directional_summary.csv", directional_summary)
    write_csv(output_dir / "rounds_per_setting.csv", rounds_rows)
    plot_rounds(rounds_rows, output_dir / "fig_rounds_lodo.pdf")


if __name__ == "__main__":
    main()
