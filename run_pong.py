"""Run the canonical fixed-round, label-free PONG protocol on cached features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from methods.pong import preprocess, refine
from utils.metrics import anmrr_score, map_score, ndcg_score


ROOT = Path(__file__).resolve().parent
FEATURES = {
    "clip": (ROOT / "data/clip_official_L14", "_ViT-L_14_official"),
    "openclip": (ROOT / "data/openclip_L14", ""),
    "dino": (ROOT / "data/dino_feats", ""),
}


def load(backbone: str, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory, suffix = FEATURES[backbone]
    query = np.load(directory / f"{dataset}_query_feats{suffix}.npy")
    target = np.load(directory / f"{dataset}_target_feats{suffix}.npy")
    query_labels = np.load(directory / f"{dataset}_query_labels{suffix}.npy").ravel().astype(int)
    target_labels = np.load(directory / f"{dataset}_target_labels{suffix}.npy").ravel().astype(int)
    return preprocess(query), preprocess(target), query_labels, target_labels


def metrics(query: np.ndarray, target: np.ndarray, query_labels: np.ndarray, target_labels: np.ndarray) -> dict[str, float]:
    distance = 1.0 - query @ target.T
    return {
        "mAP": float(map_score(distance, query_labels, target_labels)),
        "NDCG@100": float(ndcg_score(distance, query_labels, target_labels, k=100)),
        "ANMRR": float(anmrr_score(distance, query_labels, target_labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("esb", "ntu", "abo", "mn40"), required=True)
    parser.add_argument("--backbone", choices=tuple(FEATURES), required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.7)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    query, target, query_labels, target_labels = load(args.backbone, args.dataset)
    baseline = metrics(query, target, query_labels, target_labels)
    query, target = refine(query, target, k=args.k, tau=args.tau, lambda_min=args.lambda_min, lambda_max=args.lambda_max, rounds=args.rounds)
    result = metrics(query, target, query_labels, target_labels)
    print({"baseline": baseline, "pong": result, "protocol": vars(args)})


if __name__ == "__main__":
    main()
