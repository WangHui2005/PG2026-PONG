"""Small protocol-level checks that do not require the benchmark data."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.benchmark_runtime_cpu import pong_runtime
from experiments.evaluate_query_buffers import pong_refine
from experiments.evaluate_robustness import pong
from methods.config import dominant_lodo_params, lodo_grid
from methods.pong import l2_normalize, refine


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(7)
        self.query = l2_normalize(generator.normal(size=(3, 8)).astype(np.float32))
        self.target = l2_normalize(generator.normal(size=(7, 8)).astype(np.float32))
        self.params = dominant_lodo_params()

    def test_lodo_grid_and_dominant_configuration(self):
        self.assertEqual(len(lodo_grid()), 48)
        self.assertEqual(
            self.params,
            {"topk": 5, "tau": 0.2, "lam_min": 0, "lam_max": 0.7, "n_iter": 2},
        )

    def test_single_query_uses_effective_reverse_neighborhood(self):
        query, target = refine(self.query[:1], self.target, k=5, rounds=2)
        self.assertEqual(query.shape, (1, 8))
        self.assertEqual(target.shape, (7, 8))
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(target, axis=1), 1.0, rtol=1e-6)

    def test_experiment_wrappers_call_the_canonical_refinement(self):
        expected_q, expected_t = refine(
            self.query, self.target, k=5, tau=0.2, lambda_min=0, lambda_max=0.7, rounds=2
        )
        buffered_q, buffered_t = pong_refine(self.query, self.target)
        robust_q, robust_t = pong(self.query, self.target, self.params)
        np.testing.assert_allclose(buffered_q, expected_q, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(buffered_t, expected_t, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(robust_q, expected_q, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(robust_t, expected_t, rtol=1e-6, atol=1e-7)

    def test_cpu_profiler_matches_canonical_state(self):
        expected_q, expected_t = refine(
            self.query, self.target, k=5, tau=0.2, lambda_min=0, lambda_max=0.7, rounds=2
        )
        profiled_q, profiled_t, timings = pong_runtime(self.query, self.target, return_state=True)
        np.testing.assert_allclose(profiled_q, expected_q, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(profiled_t, expected_t, rtol=1e-6, atol=1e-7)
        self.assertGreater(timings["total"], 0)


if __name__ == "__main__":
    unittest.main()
