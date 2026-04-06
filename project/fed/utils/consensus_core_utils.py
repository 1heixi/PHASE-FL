from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import FitRes, parameters_to_ndarrays


def consensus_dir(working_dir: str | Path) -> Path:
    d = Path(working_dir) / "consensus_core"
    d.mkdir(parents=True, exist_ok=True)
    return d


def consensus_state_path(working_dir: str | Path) -> Path:
    return consensus_dir(working_dir) / "server_consensus_state.pt"


def load_consensus_core_state(working_dir: str | Path) -> dict[str, Any] | None:
    path = consensus_state_path(working_dir)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_consensus_core_state(
    working_dir: str | Path,
    server_round: int,
    state_dict_keys: list[str],
    ema_support_list: list[torch.Tensor],
    core_mask_list: list[torch.Tensor],
    *,
    threshold: float,
    activate_round: int,
    update_interval: int,
) -> None:
    payload = {
        "round": int(server_round),
        "state_dict_keys": list(state_dict_keys),
        "ema_support_list": [x.float().cpu() for x in ema_support_list],
        "core_mask_list": [x.bool().cpu() for x in core_mask_list],
        "threshold": float(threshold),
        "activate_round": int(activate_round),
        "update_interval": int(update_interval),
    }
    torch.save(payload, consensus_state_path(working_dir))


def _to_nonzero_tensor(arr: Any) -> torch.Tensor:
    arr_np = np.asarray(arr)
    return torch.as_tensor(arr_np != 0, dtype=torch.float32)


def extract_client_nonzero_support(results: list[tuple[Any, FitRes]]) -> list[torch.Tensor] | None:
    if len(results) == 0:
        return None

    support_sum = None
    num_clients = 0
    for _, fit_res in results:
        nds = parameters_to_ndarrays(fit_res.parameters)
        nonzero_list = [_to_nonzero_tensor(arr) for arr in nds]
        if support_sum is None:
            support_sum = [x.clone() for x in nonzero_list]
        else:
            if len(support_sum) != len(nonzero_list):
                raise ValueError("Client parameter length mismatch while building consensus support.")
            for i in range(len(support_sum)):
                if support_sum[i].shape != nonzero_list[i].shape:
                    raise ValueError(
                        f"Client parameter shape mismatch at index {i}: "
                        f"{tuple(support_sum[i].shape)} vs {tuple(nonzero_list[i].shape)}"
                    )
                support_sum[i] += nonzero_list[i]
        num_clients += 1

    if support_sum is None:
        return None
    return [x / max(num_clients, 1) for x in support_sum]


def update_and_save_consensus_core_state(
    *,
    working_dir: str | Path,
    server_round: int,
    results: list[tuple[Any, FitRes]],
    state_dict_keys: list[str],
    consensus_activate_round: int,
    consensus_ema: float,
    consensus_threshold: float,
    consensus_update_interval: int,
) -> None:
    if server_round < int(consensus_activate_round):
        return

    interval = max(int(consensus_update_interval), 1)
    if ((server_round - int(consensus_activate_round)) % interval) != 0:
        return

    current_support = extract_client_nonzero_support(results)
    if current_support is None:
        return

    if len(current_support) != len(state_dict_keys):
        raise ValueError(
            f"Consensus support length {len(current_support)} does not match state_dict_keys length {len(state_dict_keys)}"
        )

    prev = load_consensus_core_state(working_dir)
    if prev is None or "ema_support_list" not in prev or prev.get("state_dict_keys") != list(state_dict_keys):
        ema_support_list = [x.clone() for x in current_support]
    else:
        old_ema = prev["ema_support_list"]
        if len(old_ema) != len(current_support):
            ema_support_list = [x.clone() for x in current_support]
        else:
            ema_support_list = []
            for old, cur in zip(old_ema, current_support):
                old = old.float()
                cur = cur.float()
                if old.shape != cur.shape:
                    ema_support_list.append(cur.clone())
                else:
                    ema_support_list.append(consensus_ema * old + (1.0 - consensus_ema) * cur)

    core_mask_list = [x >= float(consensus_threshold) for x in ema_support_list]
    save_consensus_core_state(
        working_dir=working_dir,
        server_round=server_round,
        state_dict_keys=state_dict_keys,
        ema_support_list=ema_support_list,
        core_mask_list=core_mask_list,
        threshold=consensus_threshold,
        activate_round=consensus_activate_round,
        update_interval=consensus_update_interval,
    )
