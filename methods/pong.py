"""Canonical label-free PONG refinement."""

from __future__ import annotations

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def preprocess(view_features: np.ndarray) -> np.ndarray:
    """Mean-pool rendered views, apply tanh, then L2-normalize."""
    return l2_normalize(np.tanh(view_features.mean(axis=1).astype(np.float32)))


def _topk(similarity: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    effective_k = min(k, similarity.shape[1])
    indices = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
    return indices, np.take_along_axis(similarity, indices, axis=1)


def _weights(scores: np.ndarray, tau: float) -> np.ndarray:
    scaled = scores / tau
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def refine(
    query: np.ndarray,
    target: np.ndarray,
    *,
    k: int = 5,
    tau: float = 0.2,
    lambda_min: float = 0.0,
    lambda_max: float = 0.7,
    rounds: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exactly ``rounds`` label-free alternating PONG updates.

    Neighborhoods are recomputed before both directional updates. The target
    update uses the just-refined query features. No metric or labels are used
    to select an intermediate state.
    """
    if k < 1 or rounds < 1 or tau <= 0:
        raise ValueError("k and rounds must be positive, and tau must be positive")
    query = l2_normalize(query.astype(np.float32, copy=True))
    target = l2_normalize(target.astype(np.float32, copy=True))

    for _ in range(rounds):
        idx_q, scores_q = _topk(query @ target.T, k)
        aggregate_q = np.einsum("nk,nkd->nd", _weights(scores_q, tau), target[idx_q], optimize=True)
        confidence_q = np.clip(scores_q.mean(axis=1), 0.0, 1.0)
        lambda_q = lambda_min + (lambda_max - lambda_min) * confidence_q
        query = l2_normalize((1.0 - lambda_q[:, None]) * query + lambda_q[:, None] * aggregate_q)

        idx_t, scores_t = _topk(target @ query.T, k)
        aggregate_t = np.einsum("nk,nkd->nd", _weights(scores_t, tau), query[idx_t], optimize=True)
        confidence_t = np.clip(scores_t.mean(axis=1), 0.0, 1.0)
        lambda_t = lambda_min + (lambda_max - lambda_min) * confidence_t
        target = l2_normalize((1.0 - lambda_t[:, None]) * target + lambda_t[:, None] * aggregate_t)

    return query, target
