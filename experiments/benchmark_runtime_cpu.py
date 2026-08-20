"""CPU runtime breakdown for the reset full-batch PONG protocol.

The script uses the same frozen features and vectorized refinement used by the
LODO experiments. It reports exact-search stages separately rather
than treating propagation rounds as a proxy for wall-clock cost.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.config import dominant_lodo_params
from methods.pong import l2_normalize

RESULT_DIR = PROJECT_ROOT / "results/runtime_cpu"
FEATURE_SETS = {
    "CLIP_L14": (PROJECT_ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "OpenCLIP_L14": (PROJECT_ROOT / "data/openclip_L14", ""),
    "DINOv2": (PROJECT_ROOT / "data/dino_feats", ""),
}
DATASETS = ("esb", "ntu", "abo", "mn40")
BASE_PARAMS = dominant_lodo_params()
PARAMS = {**BASE_PARAMS, "rounds": BASE_PARAMS["n_iter"]}
WARMUPS = 2
REPEATS = 7


def softmax(scores, tau):
    values = scores / tau
    values -= values.max(axis=1, keepdims=True)
    weights = np.exp(values)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def topk(similarity, k):
    k = min(k, similarity.shape[1])
    indices = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    return indices, np.take_along_axis(similarity, indices, axis=1)


def load(backbone, dataset):
    directory, suffix = FEATURE_SETS[backbone]
    q = np.load(directory / f"{dataset}_query_feats{suffix}.npy").mean(1).astype(np.float32)
    t = np.load(directory / f"{dataset}_target_feats{suffix}.npy").mean(1).astype(np.float32)
    return l2_normalize(np.tanh(q)), l2_normalize(np.tanh(t))


def timed_add(timings, stage, started):
    timings[stage] += time.perf_counter() - started


def pong_runtime(query0, target0, return_state=False):
    """Instrumented NumPy mirror of methods.pong.refine for profiling."""
    q, t = query0.copy(), target0.copy()
    timings = {key: 0.0 for key in (
        "q_similarity", "q_topk", "q_update", "t_similarity", "t_topk", "t_update", "final_retrieval"
    )}
    for _ in range(PARAMS["rounds"]):
        started = time.perf_counter(); sim_q = q @ t.T; timed_add(timings, "q_similarity", started)
        started = time.perf_counter(); idx_q, score_q = topk(sim_q, PARAMS["topk"]); timed_add(timings, "q_topk", started)
        started = time.perf_counter()
        weights_q = softmax(score_q, PARAMS["tau"])
        agg_q = np.einsum("nk,nkd->nd", weights_q, t[idx_q], optimize=True)
        lam_q = PARAMS["lam_min"] + (PARAMS["lam_max"] - PARAMS["lam_min"]) * np.clip(score_q.mean(1), 0, 1)
        q = l2_normalize((1 - lam_q[:, None]) * q + lam_q[:, None] * agg_q)
        timed_add(timings, "q_update", started)

        started = time.perf_counter(); sim_t = t @ q.T; timed_add(timings, "t_similarity", started)
        started = time.perf_counter(); idx_t, score_t = topk(sim_t, PARAMS["topk"]); timed_add(timings, "t_topk", started)
        started = time.perf_counter()
        weights_t = softmax(score_t, PARAMS["tau"])
        agg_t = np.einsum("nk,nkd->nd", weights_t, q[idx_t], optimize=True)
        lam_t = PARAMS["lam_min"] + (PARAMS["lam_max"] - PARAMS["lam_min"]) * np.clip(score_t.mean(1), 0, 1)
        t = l2_normalize((1 - lam_t[:, None]) * t + lam_t[:, None] * agg_t)
        timed_add(timings, "t_update", started)

    started = time.perf_counter(); _ = q @ t.T; timed_add(timings, "final_retrieval", started)
    timings["total"] = sum(timings.values())
    if return_state:
        return q, t, timings
    return timings


def estimated_working_memory_mb(nq, nt, dim):
    # Float32 feature banks, one dense cross-similarity workspace, and the
    # largest gathered k-neighbor tensor used by either directional update.
    return 4 * (
        (nq + nt) * dim + nq * nt + max(nq, nt) * PARAMS["topk"] * dim
    ) / 1024 ** 2


def median_dict(items):
    return {key: float(np.median([item[key] for item in items])) for key in items[0]}


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for backbone in FEATURE_SETS:
        for dataset in DATASETS:
            query, target = load(backbone, dataset)
            for _ in range(WARMUPS):
                pong_runtime(query, target)
            runs = [pong_runtime(query, target) for _ in range(REPEATS)]
            summary = median_dict(runs)
            rows.append({
                "backbone": backbone,
                "dataset": dataset,
                "Nq": len(query),
                "Nt": len(target),
                "dimension": query.shape[1],
                "estimated_working_memory_mb": estimated_working_memory_mb(len(query), len(target), query.shape[1]),
                **{key + "_ms": value * 1000 for key, value in summary.items()},
            })
            print(backbone, dataset, f"{summary['total'] * 1000:.2f} ms", flush=True)

    with (RESULT_DIR / "per_setting.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    stage_keys = [key for key in rows[0] if key.endswith("_ms")]
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in stage_keys}
    aggregate["settings"] = len(rows)
    aggregate["warmups"] = WARMUPS
    aggregate["repeats"] = REPEATS
    aggregate["params"] = PARAMS
    with (RESULT_DIR / "metadata.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)


if __name__ == "__main__":
    main()
