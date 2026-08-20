"""Same-GPU wall-clock benchmark for PONG and TeDA test-time refinement.

Both methods receive the same preloaded CLIP-L/14 frozen features. Timings
include full descending ranking of the final score matrix, and exclude feature
extraction, loading, CPU metric computation, and labels.
PONG uses the fold-selected two-round parameters. TeDA reproduces the public
test-time classifier optimization (2000 iterations) with its dataset tau
configuration. CUDA synchronization brackets every measurement.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/runtime_gpu"
FEATURE_DIR = ROOT / "data/clip_official_L14"
SUFFIX = "_ViT-L_14_official"
DATASETS = ("esb", "ntu", "abo", "mn40")
TEDA = {
    "esb": {"tau_t": 0.03, "tau_i": 0.09},
    "ntu": {"tau_t": 0.05, "tau_i": 0.09},
    "mn40": {"tau_t": 0.03, "tau_i": 0.14},
    "abo": {"tau_t": 0.04, "tau_i": 0.09},
}
WARMUPS = 3
REPEATS = 5
TEDA_ITERATIONS = 2000


def l2(values):
    return F.normalize(values, dim=1)


def load_params():
    rows = csv.DictReader((ROOT / "results/lodo/selected_parameters.csv").open())
    return {
        row["heldout_dataset"]: {
            "topk": int(row["topk"]), "tau": float(row["tau"]),
            "lam_min": float(row["lam_min"]), "lam_max": float(row["lam_max"]),
            "n_iter": int(row["n_iter"]),
        }
        for row in rows
    }


def load_features(dataset, device):
    query = np.load(FEATURE_DIR / f"{dataset}_query_feats{SUFFIX}.npy").mean(1).astype(np.float32)
    target = np.load(FEATURE_DIR / f"{dataset}_target_feats{SUFFIX}.npy").mean(1).astype(np.float32)
    query_labels = np.load(FEATURE_DIR / f"{dataset}_query_labels{SUFFIX}.npy").ravel().astype(int)
    target_labels = np.load(FEATURE_DIR / f"{dataset}_target_labels{SUFFIX}.npy").ravel().astype(int)
    # List conversion is outside the timed region and avoids the broken
    # NumPy-2.x bridge in the archived model_align_v2 PyTorch environment.
    query = torch.tensor(np.tanh(query).tolist(), dtype=torch.float32, device=device)
    target = torch.tensor(np.tanh(target).tolist(), dtype=torch.float32, device=device)
    return l2(query), l2(target), query_labels, target_labels


def topk(similarity, size):
    return torch.topk(similarity, k=min(size, similarity.shape[1]), dim=1)


def pong(query0, target0, params, stages=None):
    query, target = query0.clone(), target0.clone()
    for _ in range(params["n_iter"]):
        start = time.perf_counter() if stages is not None else None
        sim_q = query @ target.T
        if stages is not None:
            torch.cuda.synchronize(); stages["q_similarity"] += time.perf_counter() - start
            start = time.perf_counter()
        scores_q, indices_q = topk(sim_q, params["topk"])
        if stages is not None:
            torch.cuda.synchronize(); stages["q_topk"] += time.perf_counter() - start
            start = time.perf_counter()
        weights_q = torch.softmax(scores_q / params["tau"], dim=1)
        aggregate_q = (weights_q.unsqueeze(-1) * target[indices_q]).sum(1)
        lam_q = params["lam_min"] + (params["lam_max"] - params["lam_min"]) * scores_q.mean(1).clamp(0, 1)
        query = l2((1 - lam_q[:, None]) * query + lam_q[:, None] * aggregate_q)
        if stages is not None:
            torch.cuda.synchronize(); stages["q_update"] += time.perf_counter() - start

        start = time.perf_counter() if stages is not None else None
        sim_t = target @ query.T
        if stages is not None:
            torch.cuda.synchronize(); stages["t_similarity"] += time.perf_counter() - start
            start = time.perf_counter()
        scores_t, indices_t = topk(sim_t, params["topk"])
        if stages is not None:
            torch.cuda.synchronize(); stages["t_topk"] += time.perf_counter() - start
            start = time.perf_counter()
        weights_t = torch.softmax(scores_t / params["tau"], dim=1)
        aggregate_t = (weights_t.unsqueeze(-1) * query[indices_t]).sum(1)
        lam_t = params["lam_min"] + (params["lam_max"] - params["lam_min"]) * scores_t.mean(1).clamp(0, 1)
        target = l2((1 - lam_t[:, None]) * target + lam_t[:, None] * aggregate_t)
        if stages is not None:
            torch.cuda.synchronize(); stages["t_update"] += time.perf_counter() - start
    start = time.perf_counter() if stages is not None else None
    scores = query @ target.T
    if stages is not None:
        torch.cuda.synchronize(); stages["final_retrieval"] += time.perf_counter() - start
    return scores


def teda(query0, target0, config):
    """GPU implementation of TeDA's `retrieval_eval` and `image_opt`."""
    query, target = l2(query0), l2(target0)
    initial_classifier = query.T
    pseudo = torch.softmax((target @ query.T) / config["tau_t"], dim=1)
    values, indices = pseudo.max(dim=1)
    pseudo = pseudo.clone()
    mask = values > 0.6
    pseudo[mask] = 0
    pseudo[mask, indices[mask]] = 1
    base = target.T @ pseudo
    classifier = initial_classifier.clone()
    lr, previous_norm = 10.0, float("inf")
    for _ in range(TEDA_ITERATIONS):
        probability = torch.softmax((target @ classifier) / config["tau_i"], dim=1)
        gradient = target.T @ probability - base
        gradient_norm = torch.linalg.vector_norm(gradient).item()
        if gradient_norm > previous_norm:
            lr /= 2.0
        previous_norm = gradient_norm
        classifier = classifier - (lr / (target.shape[0] * config["tau_i"])) * gradient
        classifier = F.normalize(classifier, dim=0)
    return (target @ classifier).T


def exact_map(scores, query_labels, target_labels):
    ranking = np.argsort(-scores, axis=1)
    values = []
    for index, order in enumerate(ranking):
        positions = np.flatnonzero(target_labels[order] == query_labels[index])
        if not len(positions):
            values.append(0.0); continue
        precision = np.arange(1, len(positions) + 1) / (positions + 1)
        values.append(float(np.maximum.accumulate(precision[::-1])[::-1].mean()))
    return float(np.mean(values))


def measure(method, query, target, config, record_stages=False):
    torch.cuda.reset_peak_memory_stats()
    stages = {key: 0.0 for key in ("q_similarity", "q_topk", "q_update", "t_similarity", "t_topk", "t_update", "final_retrieval", "final_ranking")}
    torch.cuda.synchronize(); start = time.perf_counter()
    scores = method(query, target, config, stages) if record_stages else method(query, target, config)
    rank_start = time.perf_counter() if record_stages else None
    _ = torch.argsort(scores, dim=1, descending=True)
    if record_stages:
        torch.cuda.synchronize(); stages["final_ranking"] += time.perf_counter() - rank_start
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    return elapsed, torch.cuda.max_memory_allocated() / 1024**2, scores, stages


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    params_by_dataset = load_params()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        query, target, ql, tl = load_features(dataset, device)
        pong_config = params_by_dataset[dataset]
        # Warm-ups include refinement, final similarity, and full ranking.
        for _ in range(WARMUPS):
            _ = torch.argsort(pong(query, target, pong_config), dim=1, descending=True)
            _ = torch.argsort(teda(query, target, TEDA[dataset]), dim=1, descending=True)
        pong_runs = [measure(pong, query, target, pong_config, True) for _ in range(REPEATS)]
        teda_runs = [measure(teda, query, target, TEDA[dataset]) for _ in range(REPEATS)]
        pong_times = np.asarray([item[0] for item in pong_runs])
        teda_times = np.asarray([item[0] for item in teda_runs])
        selected = int(np.argsort(pong_times)[len(pong_times) // 2])
        pong_scores = np.asarray(pong_runs[selected][2].float().cpu().tolist(), dtype=np.float32)
        rows.append({
            "dataset": dataset, "Nq": len(query), "Nt": len(target),
            "pong_mAP_gpu": exact_map(pong_scores, ql, tl),
            "pong_median_ms": float(np.median(pong_times) * 1000),
            "teda_median_ms": float(np.median(teda_times) * 1000),
            "speedup_teda_over_pong": float(np.median(teda_times) / np.median(pong_times)),
            "pong_peak_memory_mb": float(np.max([item[1] for item in pong_runs])),
            "teda_peak_memory_mb": float(np.max([item[1] for item in teda_runs])),
            **{f"pong_{key}_ms": value * 1000 for key, value in pong_runs[selected][3].items()},
        })
        print(dataset, rows[-1], flush=True)
    with (OUT / "wallclock.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "numpy": np.__version__,
        "conda_environment": "model_align",
        "device": torch.cuda.get_device_name(device), "dtype": "float32", "warmups": WARMUPS,
        "repeats": REPEATS, "teda_iterations": TEDA_ITERATIONS,
        "timed_scope": "refinement, final similarity, and full descending ranking; excludes feature extraction, loading, CPU metrics, and labels",
        "mean_pong_ms": float(np.mean([r["pong_median_ms"] for r in rows])),
        "mean_teda_ms": float(np.mean([r["teda_median_ms"] for r in rows])),
        "mean_speedup": float(np.mean([r["speedup_teda_over_pong"] for r in rows])),
    }
    with (OUT / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
