"""Configuration helpers for PONG experiments."""

from __future__ import annotations

import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LODO_PROTOCOL_PATH = ROOT / "configs" / "lodo_protocol.json"


def load_lodo_protocol() -> dict:
    with LODO_PROTOCOL_PATH.open() as handle:
        return json.load(handle)


def lodo_grid() -> list[dict[str, float | int]]:
    """Return the 48 configurations stored in the formal LODO protocol."""
    space = load_lodo_protocol()["candidate_space"]
    return [
        {
            "topk": k,
            "tau": tau,
            "lam_min": lambda_min,
            "lam_max": lambda_max,
            "n_iter": rounds,
        }
        for k, tau, lambda_min, lambda_max, rounds in itertools.product(
            space["k"],
            space["tau"],
            space["lambda_min"],
            space["lambda_max"],
            space["rounds"],
        )
    ]


def dominant_lodo_params() -> dict[str, float | int]:
    """Return the dominant shared configuration in experiment-script names."""
    values = load_lodo_protocol()["dominant_shared_configuration"]
    return {
        "topk": values["k"],
        "tau": values["tau"],
        "lam_min": values["lambda_min"],
        "lam_max": values["lambda_max"],
        "n_iter": values["rounds"],
    }
