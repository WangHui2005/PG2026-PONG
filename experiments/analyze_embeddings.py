"""Generate LODO-consistent mechanism and qualitative analysis artifacts.

All quantitative refinements use the configuration selected for the held-out
dataset by the formal LODO protocol. No held-out metric selects a parameter,
round, intermediate state, class, or reported configuration. Labels are used
only after refinement to compute diagnostic metrics and color the post-hoc
t-SNE visualization.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.pong import l2_normalize, refine


FEATURE_SETS = {
    "CLIP_L14": (PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
DISPLAY_DATASETS = {
    "esb": "OS-ESB-core",
    "ntu": "OS-NTU-core",
    "abo": "OS-ABO-core",
    "mn40": "OS-MN40-core",
}
OUT_DIR = PROJECT_ROOT / "results/embedding_analysis"


def publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fold_params() -> dict[str, dict[str, float | int]]:
    selected = {}
    for row in read_csv(PROJECT_ROOT / "results/lodo/selected_parameters.csv"):
        selected[row["heldout_dataset"]] = {
            "k": int(row["topk"]),
            "tau": float(row["tau"]),
            "lambda_min": float(row["lam_min"]),
            "lambda_max": float(row["lam_max"]),
            "rounds": int(row["n_iter"]),
        }
    return selected


def load_setting(backbone: str, dataset: str):
    feature_dir, suffix = FEATURE_SETS[backbone]
    query = np.load(feature_dir / f"{dataset}_query_feats{suffix}.npy").mean(1).astype(np.float32)
    target = np.load(feature_dir / f"{dataset}_target_feats{suffix}.npy").mean(1).astype(np.float32)
    query_labels = np.load(feature_dir / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(feature_dir / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return l2_normalize(np.tanh(query)), l2_normalize(np.tanh(target)), query_labels, target_labels


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_gain_figure():
    rows = read_csv(PROJECT_ROOT / "results/ablations/directional_summary.csv")
    labels = {
        "baseline": "Baseline",
        "query_only": "Q-only",
        "target_only": "T-only",
        "simultaneous": "Simul.",
        "alternating_fixed": "Altern.",
        "pong_adaptive": "PONG",
    }
    colors = ["#EAF5FF", "#B9DDF7", "#88C9F2", "#55B3EC", "#299BE1", "#1565B3"]
    gains = [float(row["delta_over_baseline_pp"]) for row in rows]
    with plt.rc_context({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.55, 1.0]})
        ax = axes[0]
        x = np.arange(len(rows))
        ax.bar(x, gains, width=0.58, color=colors, edgecolor="white", linewidth=0.4)
        for index, gain in enumerate(gains):
            if gain > 0:
                ax.text(index, gain + 0.14, f"+{gain:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x, [labels[row["variant"]] for row in rows])
        ax.set_ylabel("$\\Delta$mAP over baseline (pp)")
        ax.set_ylim(0, max(gains) + 1.05)
        ax.set_title("(a) Ablation: Absolute Gains by Variant", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E4EEF6", linewidth=0.6)
        ax.set_axisbelow(True)
        despine(ax)

        ax = axes[1]
        stage_labels = ["Base", "+Q\nref.", "+Bi-\ndir.", "+Alt.", "+ANA", "PONG"]
        increments = [0.0, gains[1], gains[3] - gains[1], gains[4] - gains[3], gains[5] - gains[4]]
        cumulative = np.cumsum(increments)
        bottoms = [0.0, 0.0, cumulative[1], cumulative[2], cumulative[3], 0.0]
        heights = [0.0, increments[1], increments[2], increments[3], increments[4], gains[5]]
        stage_colors = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[5]]
        sx = np.arange(len(stage_labels))
        ax.bar(sx, heights, bottom=bottoms, width=0.55, color=stage_colors, edgecolor="white", linewidth=0.4)
        for index in range(1, 5):
            level = cumulative[index]
            ax.plot([index + 0.275, index + 0.725], [level, level], color="#8DA9BC", linestyle="--", linewidth=0.6)
            ax.text(index, level + 0.14, f"+{increments[index]:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(5, gains[5] + 0.14, f"{gains[5]:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1565B3")
        ax.set_xticks(sx, stage_labels)
        ax.tick_params(axis="x", labelsize=7.5, pad=2)
        ax.set_ylabel("Cumulative $\\Delta$mAP (pp)")
        ax.set_ylim(0, max(gains) + 1.05)
        ax.set_title("(b) Gain Decomposition", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E4EEF6", linewidth=0.6)
        ax.set_axisbelow(True)
        despine(ax)
        fig.tight_layout(pad=0.6, w_pad=1.1)
    fig.savefig(OUT_DIR / "fig_gain_decomp_lodo.pdf")
    plt.close(fig)


def generate_development_sensitivity_figure():
    rows = read_csv(PROJECT_ROOT / "results/lodo/development_grid_scores.csv")
    fold_order = ("esb", "ntu", "abo", "mn40")
    colors = {"esb": "#4477AA", "ntu": "#228833", "abo": "#CC6677", "mn40": "#AA3377"}
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.0))
    dimensions = (("topk", [5, 7, 10, 15], "$k$"), ("lam_max", [0.5, 0.7, 0.9], "$\\lambda_{\\max}$"))
    for ax, (column, values, xlabel) in zip(axes, dimensions):
        for fold in fold_order:
            means = []
            for value in values:
                selected = [
                    float(row["development_mean_mAP"])
                    for row in rows
                    if row["heldout_dataset"] == fold and float(row[column]) == float(value)
                ]
                means.append(100.0 * float(np.mean(selected)))
            ax.plot(values, means, marker="o", markersize=3.0, color=colors[fold], label=DISPLAY_DATASETS[fold].replace("OS-", ""))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Development mean mAP (%)")
        ax.set_xticks(values)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.4)
        ax.set_axisbelow(True)
        despine(ax)
    axes[0].set_title("(a) Neighborhood size", loc="left", fontweight="bold")
    axes[1].set_title("(b) Blending bound", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout(pad=0.5, h_pad=0.8)
    fig.savefig(OUT_DIR / "fig_sensitivity_lodo.pdf")
    plt.close(fig)


def topk_indices(query: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    similarity = query @ target.T
    effective_k = min(k, similarity.shape[1])
    return np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]


def neighborhood_purity(indices: np.ndarray, query_labels: np.ndarray, target_labels: np.ndarray) -> float:
    return float(np.mean(target_labels[indices] == query_labels[:, None]))


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64))
    total = values.sum()
    if total == 0:
        return 0.0
    n = len(values)
    return float((2.0 * np.dot(np.arange(1, n + 1), values) / total - (n + 1)) / n)


def hubness_gini(indices: np.ndarray, number_of_targets: int) -> float:
    counts = np.bincount(indices.ravel(), minlength=number_of_targets)
    return gini(counts)


def fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean inter-class cosine distance divided by mean intra-class distance."""
    class_sums = {}
    class_sizes = {}
    intra = []
    for label in np.unique(labels):
        selected = features[labels == label]
        n = len(selected)
        if n < 2:
            continue
        vector_sum = selected.sum(axis=0, dtype=np.float64)
        mean_similarity = (float(vector_sum @ vector_sum) - n) / (n * (n - 1))
        intra.append(1.0 - mean_similarity)
        class_sums[int(label)] = vector_sum
        class_sizes[int(label)] = n
    class_labels = sorted(class_sums)
    inter = []
    for i, label_i in enumerate(class_labels):
        for label_j in class_labels[i + 1:]:
            mean_similarity = float(class_sums[label_i] @ class_sums[label_j]) / (class_sizes[label_i] * class_sizes[label_j])
            inter.append(1.0 - mean_similarity)
    return float(np.mean(inter) / max(float(np.mean(intra)), 1e-12))


def generate_geometry_metrics(fold_params):
    rows = []
    for dataset in DATASETS:
        params = fold_params[dataset]
        for backbone in FEATURE_SETS:
            query, target, query_labels, target_labels = load_setting(backbone, dataset)
            refined_query, refined_target = refine(query, target, **params)
            before_indices = topk_indices(query, target, params["k"])
            after_indices = topk_indices(refined_query, refined_target, params["k"])
            row = {
                "heldout_dataset": dataset,
                "backbone": backbone,
                **params,
                "purity_before": neighborhood_purity(before_indices, query_labels, target_labels),
                "purity_after": neighborhood_purity(after_indices, query_labels, target_labels),
                "hubness_gini_before": hubness_gini(before_indices, len(target)),
                "hubness_gini_after": hubness_gini(after_indices, len(target)),
                "fisher_before": fisher_ratio(target, target_labels),
                "fisher_after": fisher_ratio(refined_target, target_labels),
            }
            rows.append(row)
            print(f"Geometry complete: {backbone} / {dataset}", flush=True)
    write_csv(OUT_DIR / "geometry_per_setting.csv", rows)
    summary = {}
    for metric in ("purity", "hubness_gini", "fisher"):
        deltas = [row[f"{metric}_after"] - row[f"{metric}_before"] for row in rows]
        summary[metric] = {
            "before_mean": float(np.mean([row[f"{metric}_before"] for row in rows])),
            "after_mean": float(np.mean([row[f"{metric}_after"] for row in rows])),
            "delta_mean": float(np.mean(deltas)),
            "improved_settings": int(sum(delta > 0 for delta in deltas)) if metric != "hubness_gini" else int(sum(delta < 0 for delta in deltas)),
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas)),
        }
    return summary


def balanced_indices(labels: np.ndarray, classes: list[int], per_class: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    indices = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) > per_class:
            candidates = generator.choice(candidates, size=per_class, replace=False)
        indices.extend(candidates.tolist())
    return np.asarray(indices, dtype=int)


def generate_tsne(fold_params):
    dataset = "esb"
    params = fold_params[dataset]
    backbones = ("OpenCLIP_L14", "CLIP_L14")
    palette = plt.get_cmap("tab10").colors[:8]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.2))
    for row_index, backbone in enumerate(backbones):
        query, target, query_labels, target_labels = load_setting(backbone, dataset)
        refined_query, refined_target = refine(query, target, **params)
        classes = [int(label) for label, _ in Counter(target_labels).most_common(8)]
        target_indices = balanced_indices(target_labels, classes, per_class=80, seed=42)
        query_indices = balanced_indices(query_labels, classes, per_class=40, seed=43)
        before = np.vstack([target[target_indices], query[query_indices]])
        after = np.vstack([refined_target[target_indices], refined_query[query_indices]])
        embedded = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (len(before) * 2 - 1) // 10)),
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            random_state=42,
        ).fit_transform(np.vstack([before, after]))
        split = len(before)
        before_2d, after_2d = embedded[:split], embedded[split:]
        n_target = len(target_indices)
        target_colors = [palette[classes.index(int(label))] for label in target_labels[target_indices]]
        query_colors = [palette[classes.index(int(label))] for label in query_labels[query_indices]]
        for column, (values, title) in enumerate(((before_2d, "Before PONG"), (after_2d, "After PONG"))):
            ax = axes[row_index, column]
            ax.scatter(values[:n_target, 0], values[:n_target, 1], c=target_colors, s=4.5, alpha=0.6, marker="o", linewidths=0)
            ax.scatter(values[n_target:, 0], values[n_target:, 1], c=query_colors, s=13, alpha=0.95, marker="*", edgecolors="#222222", linewidths=0.2)
            display_backbone = backbone.replace("_L14", "")
            ax.set_title(f"{display_backbone}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#999999")
    fig.tight_layout(pad=0.35, h_pad=0.45, w_pad=0.35)
    fig.savefig(OUT_DIR / "tsne_esb_compare_lodo.pdf")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    publication_style()
    fold_params = load_fold_params()
    generate_gain_figure()
    generate_development_sensitivity_figure()
    geometry_summary = generate_geometry_metrics(fold_params)
    generate_tsne(fold_params)
    with (OUT_DIR / "metadata.json").open("w") as handle:
        json.dump({
            "protocol": "fold-selected LODO parameters, fixed final state at R=2",
            "diagnostic_label_use": "metrics and t-SNE coloring only; never parameter or iteration selection",
            "geometry_summary": geometry_summary,
        }, handle, indent=2)
    print(json.dumps(geometry_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
