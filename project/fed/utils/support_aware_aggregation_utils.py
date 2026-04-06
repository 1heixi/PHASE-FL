"""Utilities for server-side Soft Support-Aware Aggregation on prunable weights only.

This module assumes the project's true parameter exchange order:
`generic_get_parameters/generic_set_parameters` operate on parameters in
`sorted(net.state_dict().items(), key=lambda x: x[0])` order.

Only tensors marked as `prunable` will receive support-aware soft aggregation.
All other tensors keep the original strategy aggregate unchanged.

Additionally, this module exports a debiased layer-confidence map keyed by the
sorted state key names. The local third innovation can then use these scalar
signals to softly reweight pruning scores while still preserving global Top-K.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import NDArrays, parameters_to_ndarrays


def _support_state_dir(working_dir: Path) -> Path:
    path = working_dir / "support_agg"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _support_state_path(working_dir: Path) -> Path:
    return _support_state_dir(working_dir) / "support_ema_state.pt"


def _layer_confidence_state_path(working_dir: Path) -> Path:
    return _support_state_dir(working_dir) / "layer_confidence_state.pt"


def load_support_ema_state(
    working_dir: Path,
    reference_arrays: NDArrays,
) -> tuple[list[np.ndarray], int]:
    path = _support_state_path(working_dir)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = payload.get("support_ema_state", []) if isinstance(payload, dict) else []
        num_updates = int(payload.get("num_updates", 0)) if isinstance(payload, dict) else 0

        out: list[np.ndarray] = []
        for ref, raw in zip(reference_arrays, raw_state, strict=False):
            arr = np.asarray(raw, dtype=np.float32)
            ref_arr = np.asarray(ref)
            if arr.shape == ref_arr.shape:
                out.append(arr)
            else:
                out.append(np.zeros_like(ref_arr, dtype=np.float32))

        if len(out) < len(reference_arrays):
            out.extend([np.zeros_like(np.asarray(ref), dtype=np.float32) for ref in reference_arrays[len(out):]])

        return out, num_updates

    return [np.zeros_like(np.asarray(ref), dtype=np.float32) for ref in reference_arrays], 0


def save_support_ema_state(
    working_dir: Path,
    support_ema_state: list[np.ndarray],
    server_round: int,
    num_updates: int,
) -> None:
    path = _support_state_path(working_dir)
    torch.save(
        {
            "round": int(server_round),
            "num_updates": int(num_updates),
            "support_ema_state": [torch.as_tensor(np.asarray(x, dtype=np.float32)) for x in support_ema_state],
        },
        path,
    )


def save_layer_confidence_state(
    working_dir: Path,
    layer_confidence_map: dict[str, float],
    server_round: int,
    num_updates: int,
) -> None:
    path = _layer_confidence_state_path(working_dir)
    torch.save(
        {
            "round": int(server_round),
            "num_updates": int(num_updates),
            "layer_confidence_map": {str(k): float(v) for k, v in layer_confidence_map.items()},
        },
        path,
    )


def _debiased_support_prob(
    support_ema: np.ndarray,
    support_ema_beta: float,
    num_updates: int,
) -> np.ndarray:
    ema = np.asarray(support_ema, dtype=np.float32)
    if num_updates <= 0:
        return np.clip(ema, 0.0, 1.0)

    beta = float(np.clip(support_ema_beta, 0.0, 0.999999))
    denom = 1.0 - (beta ** num_updates)
    if denom <= 1e-12:
        return np.clip(ema, 0.0, 1.0)

    debiased = ema / denom
    return np.clip(debiased, 0.0, 1.0)


def _confidence_from_prob(prob: np.ndarray) -> float:
    # High when support is confidently near 0 or near 1; low near 0.5.
    return float(np.mean(np.abs(2.0 * prob - 1.0)))


def _support_aware_float_aggregate(
    prev_global: np.ndarray,
    baseline_aggregated: np.ndarray,
    client_arrays: list[np.ndarray],
    client_weights: list[float],
    prev_support_ema: np.ndarray,
    support_ema_beta: float,
    support_gamma: float,
    support_alpha_min: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prev = np.asarray(prev_global, dtype=np.float32)
    base = np.asarray(baseline_aggregated, dtype=np.float32)
    arrays = [np.asarray(a, dtype=np.float32) for a in client_arrays]
    weights = np.asarray(client_weights, dtype=np.float32)

    total_weight = float(weights.sum())
    if total_weight <= 0:
        metrics = {
            "support_mean": 0.0,
            "support_ema_mean": float(np.asarray(prev_support_ema).mean()) if prev_support_ema.size else 0.0,
            "support_alpha_mean": 0.0,
            "support_nonzero_fraction": 0.0,
            "support_prunable_fraction": 1.0,
        }
        return base.astype(prev_global.dtype, copy=False), np.asarray(prev_support_ema, dtype=np.float32), metrics

    nonzero_masks = [np.not_equal(a, 0.0).astype(np.float32) for a in arrays]

    support = np.zeros_like(prev, dtype=np.float32)
    for mask, w in zip(nonzero_masks, weights, strict=False):
        support += mask * float(w)
    support /= total_weight

    if prev_support_ema.shape != prev.shape:
        prev_support_ema = np.zeros_like(prev, dtype=np.float32)

    support_ema = support_ema_beta * prev_support_ema + (1.0 - support_ema_beta) * support

    denom_nonzero = np.zeros_like(prev, dtype=np.float32)
    num_nonzero = np.zeros_like(prev, dtype=np.float32)
    for arr, mask, w in zip(arrays, nonzero_masks, weights, strict=False):
        weighted_mask = mask * float(w)
        denom_nonzero += weighted_mask
        num_nonzero += arr * weighted_mask

    has_nonzero = denom_nonzero > 0
    avg_nonzero = base.copy()
    avg_nonzero[has_nonzero] = num_nonzero[has_nonzero] / denom_nonzero[has_nonzero]

    alpha = support_alpha_min + (1.0 - support_alpha_min) * np.power(support_ema, support_gamma)
    alpha = np.clip(alpha, support_alpha_min, 1.0).astype(np.float32)

    out = base.copy()
    out[has_nonzero] = (1.0 - alpha[has_nonzero]) * prev[has_nonzero] + alpha[has_nonzero] * avg_nonzero[has_nonzero]

    metrics = {
        "support_mean": float(support.mean()),
        "support_ema_mean": float(support_ema.mean()),
        "support_alpha_mean": float(alpha.mean()),
        "support_nonzero_fraction": float(has_nonzero.mean()),
        "support_prunable_fraction": 1.0,
    }
    return out.astype(prev_global.dtype, copy=False), support_ema.astype(np.float32), metrics


def support_aware_aggregate_from_results_prunable_only(
    *,
    server_round: int,
    prev_global_arrays: NDArrays,
    baseline_aggregated_arrays: NDArrays,
    results: list[tuple[Any, Any]],
    working_dir: Path,
    support_agg_activate_round: int,
    support_ema_beta: float,
    support_gamma: float,
    support_alpha_min: float,
    support_update_interval: int,
    prunable_flags: list[bool],
    sorted_state_keys: list[str] | None = None,
) -> tuple[NDArrays | None, dict[str, float]]:
    """Soft support-aware aggregation, applied only to prunable weights.

    Non-prunable tensors are kept exactly as produced by the original strategy
    aggregation (`baseline_aggregated_arrays`).
    """
    metrics: dict[str, float] = {
        "support_agg_active": 0.0,
        "support_mean": 0.0,
        "support_ema_mean": 0.0,
        "support_alpha_mean": 0.0,
        "support_nonzero_fraction": 0.0,
        "support_prunable_fraction": float(np.mean(prunable_flags)) if prunable_flags else 0.0,
        "support_num_prunable_tensors": float(sum(bool(x) for x in prunable_flags)),
    }

    if not results:
        return None, metrics

    interval = max(int(support_update_interval), 1)
    is_active = server_round >= int(support_agg_activate_round) and (
        (server_round - int(support_agg_activate_round)) % interval == 0
    )
    if not is_active:
        return None, metrics

    if len(prunable_flags) != len(prev_global_arrays):
        raise ValueError(
            f"prunable_flags length mismatch: {len(prunable_flags)} vs {len(prev_global_arrays)}"
        )

    if sorted_state_keys is not None and len(sorted_state_keys) != len(prev_global_arrays):
        raise ValueError(
            f"sorted_state_keys length mismatch: {len(sorted_state_keys)} vs {len(prev_global_arrays)}"
        )

    client_weights = [float(fit_res.num_examples) for _, fit_res in results]
    client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

    support_ema_state, prev_num_updates = load_support_ema_state(working_dir, prev_global_arrays)
    new_num_updates = int(prev_num_updates) + 1

    out_arrays: list[np.ndarray] = [np.asarray(arr).copy() for arr in baseline_aggregated_arrays]
    new_support_state: list[np.ndarray] = [np.zeros_like(np.asarray(ref), dtype=np.float32) for ref in prev_global_arrays]

    mean_support: list[float] = []
    mean_support_ema: list[float] = []
    mean_support_alpha: list[float] = []
    mean_support_nonzero: list[float] = []

    layer_confidence_map: dict[str, float] = {}

    for idx, (prev_arr, base_arr, is_prunable) in enumerate(
        zip(prev_global_arrays, baseline_aggregated_arrays, prunable_flags, strict=False)
    ):
        prev_np = np.asarray(prev_arr)
        base_np = np.asarray(base_arr)

        if not is_prunable or not np.issubdtype(prev_np.dtype, np.floating):
            out_arrays[idx] = base_np.astype(prev_np.dtype, copy=False)
            new_support_state[idx] = np.zeros_like(np.asarray(support_ema_state[idx]), dtype=np.float32)
            continue

        per_client = [np.asarray(arrs[idx]) for arrs in client_ndarrays]
        prev_state = np.asarray(support_ema_state[idx], dtype=np.float32)

        out, state, local_metrics = _support_aware_float_aggregate(
            prev_global=prev_np,
            baseline_aggregated=base_np,
            client_arrays=per_client,
            client_weights=client_weights,
            prev_support_ema=prev_state,
            support_ema_beta=float(support_ema_beta),
            support_gamma=float(support_gamma),
            support_alpha_min=float(support_alpha_min),
        )

        out_arrays[idx] = out
        new_support_state[idx] = state

        mean_support.append(local_metrics["support_mean"])
        mean_support_ema.append(local_metrics["support_ema_mean"])
        mean_support_alpha.append(local_metrics["support_alpha_mean"])
        mean_support_nonzero.append(local_metrics["support_nonzero_fraction"])

        if sorted_state_keys is not None:
            debiased_prob = _debiased_support_prob(
                support_ema=state,
                support_ema_beta=float(support_ema_beta),
                num_updates=new_num_updates,
            )
            layer_confidence_map[str(sorted_state_keys[idx])] = _confidence_from_prob(debiased_prob)

    save_support_ema_state(
        working_dir=working_dir,
        support_ema_state=new_support_state,
        server_round=server_round,
        num_updates=new_num_updates,
    )

    if sorted_state_keys is not None:
        save_layer_confidence_state(
            working_dir=working_dir,
            layer_confidence_map=layer_confidence_map,
            server_round=server_round,
            num_updates=new_num_updates,
        )

    metrics.update(
        {
            "support_agg_active": 1.0,
            "support_mean": float(np.mean(mean_support)) if mean_support else 0.0,
            "support_ema_mean": float(np.mean(mean_support_ema)) if mean_support_ema else 0.0,
            "support_alpha_mean": float(np.mean(mean_support_alpha)) if mean_support_alpha else 0.0,
            "support_nonzero_fraction": float(np.mean(mean_support_nonzero)) if mean_support_nonzero else 0.0,
        }
    )
    return out_arrays, metrics
