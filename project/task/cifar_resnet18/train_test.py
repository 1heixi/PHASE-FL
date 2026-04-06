"""CIFAR training and testing functions for SparsyFed.

Current trunk + Boundary-Band Hysteresis Pruning
------------------------------------------------
1. Keep the user's current cosine sparsity + GGMP/FedMCR-compatible trunk.
2. Keep prunable-name helper for server-side prunable-only support aggregation.
3. Replace the failed hard layer-allocation line with boundary-band hysteresis
   around the global Top-K threshold.
4. Preserve the exact global keep budget by still performing the final pruning
   through a single global Top-K on hysteresis-adjusted scores.
5. Do not change client/server/dispatch parameter exchange semantics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sized
from pathlib import Path
from typing import Optional, cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.task.cifar_resnet18.models import (
    get_parameters_to_prune,
    set_spectral_global_exponent,
)


class TrainConfig(BaseModel):
    """Training configuration for the current trunk + boundary-band hysteresis pruning."""

    device: torch.device
    epochs: int
    learning_rate: float
    final_learning_rate: float
    curr_round: int
    tot_rounds: int

    # Current trunk fields kept for compatibility
    ggmp_lambda: float = 0.0
    fedmcr_beta: float = 0.0
    target_sparsity: float = 0.98
    initial_sparsity: float = 0.30
    ramp_end: float = 0.56

    # Existing server-side support aggregation controls (kept for compatibility)
    support_agg_activate_round: int = 700
    support_ema_beta: float = 0.90
    support_gamma: float = 2.0
    support_alpha_min: float = 0.05
    support_update_interval: int = 1

    # Third innovation: Boundary-Band Hysteresis Pruning
    hysteresis_enable: bool = False
    hysteresis_activate_round: int = 560
    hysteresis_band_ratio: float = 0.05
    hysteresis_keep_bias: float = 0.75
    hysteresis_grow_penalty: float = 0.75
    hysteresis_min_threshold: float = 1.0e-12

    class Config:
        arbitrary_types_allowed = True


class TestConfig(BaseModel):
    device: torch.device

    class Config:
        arbitrary_types_allowed = True


def get_prunable_param_names_for_server(net: nn.Module) -> list[str]:
    """Return prunable parameter names aligned with net.named_parameters().

    This helper does NOT change training behavior. It only exposes the same
    prunable parameter set already used by local pruning logic so the server can
    build correct prunable_flags.
    """
    parameters_to_prune = get_parameters_to_prune(net)
    ptr_to_name = {p.data_ptr(): n for n, p in net.named_parameters()}

    prunable_names: list[str] = []
    seen: set[str] = set()

    for item in parameters_to_prune:
        if len(item) < 2:
            raise RuntimeError(f"Unexpected pruning spec item: {item!r}")

        module = item[0]
        param_name = item[1]

        param = getattr(module, param_name, None)
        if param is None:
            raise RuntimeError(
                f"Failed to resolve prunable parameter '{param_name}' from module {module.__class__.__name__}"
            )

        full_name = ptr_to_name.get(param.data_ptr())
        if full_name is None:
            raise RuntimeError(
                f"Prunable tensor '{param_name}' from module {module.__class__.__name__} "
                "is not present in net.named_parameters() mapping."
            )

        if full_name not in seen:
            seen.add(full_name)
            prunable_names.append(full_name)

    if len(prunable_names) == 0:
        raise RuntimeError(
            "No prunable parameter names were resolved from get_parameters_to_prune(net)."
        )

    return prunable_names


def _round_progress(curr_round: int, tot_rounds: int) -> float:
    return min(curr_round / max(tot_rounds, 1), 1.0)


def _apply_boundary_hysteresis(
    score: torch.Tensor,
    prev_mask: torch.Tensor,
    raw_threshold: float,
    band: float,
    keep_bias: float,
    grow_penalty: float,
) -> tuple[torch.Tensor, int, int]:
    """Apply hysteresis only to elements inside the threshold boundary band.

    Parameters
    ----------
    score : torch.Tensor
        Raw pruning score tensor.
    prev_mask : torch.Tensor
        Previous-round binary mask (bool tensor).
    raw_threshold : float
        Global threshold computed from raw scores.
    band : float
        Half-width of the threshold boundary band.
    keep_bias : float
        Positive bias multiplier for previously-active boundary elements.
    grow_penalty : float
        Negative bias multiplier for previously-inactive boundary elements.

    Returns
    -------
    adjusted_score : torch.Tensor
        Score after local hysteresis adjustment.
    boundary_count : int
        Number of elements that fell into the boundary band.
    prev_keep_boundary_count : int
        Number of boundary-band elements that were previously active.
    """
    if band <= 0.0:
        return score, 0, 0

    boundary = torch.abs(score - raw_threshold) <= band
    if not torch.any(boundary):
        return score, 0, 0

    adjusted = score.clone()

    prev_keep_boundary = boundary & prev_mask
    prev_prune_boundary = boundary & (~prev_mask)

    if keep_bias != 0.0:
        adjusted[prev_keep_boundary] += band * float(keep_bias)
    if grow_penalty != 0.0:
        adjusted[prev_prune_boundary] -= band * float(grow_penalty)

    adjusted = torch.clamp(adjusted, min=0.0)

    boundary_count = int(boundary.sum().item())
    prev_keep_boundary_count = int(prev_keep_boundary.sum().item())
    return adjusted, boundary_count, prev_keep_boundary_count


def train(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Standard unpruned baseline training."""
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader can't be 0, exiting...")

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, weight_decay=0.001)

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()
    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def fixed_train(
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    global_masks: Optional[dict[str, torch.Tensor]] = None,
    dynamic_beta: float = 0.0,
) -> tuple[int, dict]:
    """Train with Dynamic FedMCR (SSE approach)."""
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader can't be 0, exiting...")

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, weight_decay=0.001)

    fedmcr_params = []
    use_fedmcr = dynamic_beta > 0.0 and global_masks is not None

    if use_fedmcr:
        for name, param in net.named_parameters():
            if name in global_masks:
                m_global = global_masks[name].to(config.device, non_blocking=True)
                p_mask = (1.0 - m_global)
                fedmcr_params.append((param, p_mask))

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    final_reg_loss = 0.0

    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        final_reg_loss = 0.0

        for data, target in trainloader:
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)

            if use_fedmcr:
                reg_loss = 0.0
                for param, p_mask in fedmcr_params:
                    reg_loss += torch.sum((param ** 2) * p_mask)
                loss += dynamic_beta * reg_loss
                final_reg_loss += reg_loss.item()

            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
        "fedmcr_reg_loss": final_reg_loss / len(cast(Sized, trainloader.dataset)),
    }


def get_fixed_train_and_prune(
    alpha: float = 1.0, amount: float = 0.0, pruning_method: str = "l1"
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Current trunk + boundary-band hysteresis pruning.

    The final keep budget remains exact because pruning is still decided by a
    single global Top-K on the adjusted scores. The hysteresis module only acts
    as a boundary-aware tie-breaker near the current global threshold.
    """
    del alpha, amount, pruning_method

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        curr_round = int(_config.get("curr_round", 1))
        tot_rounds = int(_config.get("tot_rounds", 1000))
        config = TrainConfig(**_config)

        progress = _round_progress(curr_round, tot_rounds)

        init_sparsity = config.initial_sparsity
        target_sparsity = config.target_sparsity
        ramp_end = float(config.ramp_end)

        if progress >= ramp_end:
            current_sparsity = target_sparsity
            current_lambda = 0.0
            current_beta = config.fedmcr_beta
        else:
            u = progress / max(ramp_end, 1e-8)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * u))
            current_sparsity = target_sparsity + (init_sparsity - target_sparsity) * cosine_decay
            current_lambda = config.ggmp_lambda * cosine_decay
            current_beta = config.fedmcr_beta * (u ** 3)

        global_masks: dict[str, torch.Tensor] = {}
        initial_net_state: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in net.named_parameters():
                initial_net_state[name] = param.detach().clone().cpu()
                global_masks[name] = (param.detach().abs() > 1e-4).float().cpu()

        num_samples, metrics = fixed_train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
            global_masks=global_masks,
            dynamic_beta=current_beta,
        )

        total_iou = total_changed = total_new_regrown = total_pruned = 0.0
        total_prev_one = total_numel = total_keep_target = 0.0
        layer_count = 0

        hysteresis_active = False
        hysteresis_threshold = 0.0
        hysteresis_band = 0.0
        total_boundary_count = 0
        total_prev_keep_boundary_count = 0

        if current_sparsity > 0.0:
            parameters_to_prune = list(get_parameters_to_prune(net))
            ptr_to_name = {p.data_ptr(): n for n, p in net.named_parameters()}

            raw_score_cache: dict[str, torch.Tensor] = {}
            prev_mask_cache: dict[str, torch.Tensor] = {}
            all_raw_scores = []

            with torch.no_grad():
                for module, name, _ in parameters_to_prune:
                    weight = getattr(module, name)
                    param_name = ptr_to_name[weight.data_ptr()]
                    initial_w_gpu = initial_net_state[param_name].to(weight.device)
                    delta_w = weight.detach() - initial_w_gpu

                    raw_score = torch.abs(weight.detach()) + current_lambda * torch.abs(delta_w)
                    raw_score = raw_score + 1e-9 * torch.rand_like(raw_score)

                    raw_score_cache[param_name] = raw_score
                    prev_mask_cache[param_name] = global_masks[param_name].to(weight.device).bool()
                    all_raw_scores.append(raw_score.view(-1))

                all_raw_scores_flat = torch.cat(all_raw_scores)
                total_num_elements = int(all_raw_scores_flat.numel())
                total_keep = int((1.0 - current_sparsity) * total_num_elements)
                total_keep = min(max(total_keep, 1), total_num_elements)

                raw_threshold_tensor, _ = torch.topk(all_raw_scores_flat, total_keep, sorted=True)
                hysteresis_threshold = float(raw_threshold_tensor[-1].item())

                if (
                    bool(config.hysteresis_enable)
                    and progress >= ramp_end
                    and curr_round >= int(config.hysteresis_activate_round)
                ):
                    base_thr = max(hysteresis_threshold, float(config.hysteresis_min_threshold))
                    hysteresis_band = float(config.hysteresis_band_ratio) * base_thr
                    hysteresis_active = hysteresis_band > 0.0

                adjusted_score_cache: dict[str, torch.Tensor] = {}
                all_adjusted_scores = []

                for module, name, _ in parameters_to_prune:
                    weight = getattr(module, name)
                    param_name = ptr_to_name[weight.data_ptr()]
                    raw_score = raw_score_cache[param_name]
                    prev_mask = prev_mask_cache[param_name]

                    if hysteresis_active:
                        adjusted_score, boundary_count, prev_keep_boundary_count = _apply_boundary_hysteresis(
                            score=raw_score,
                            prev_mask=prev_mask,
                            raw_threshold=hysteresis_threshold,
                            band=hysteresis_band,
                            keep_bias=float(config.hysteresis_keep_bias),
                            grow_penalty=float(config.hysteresis_grow_penalty),
                        )
                        total_boundary_count += boundary_count
                        total_prev_keep_boundary_count += prev_keep_boundary_count
                    else:
                        adjusted_score = raw_score

                    adjusted_score_cache[param_name] = adjusted_score
                    all_adjusted_scores.append(adjusted_score.view(-1))

                all_adjusted_scores_flat = torch.cat(all_adjusted_scores)
                final_threshold_tensor, _ = torch.topk(all_adjusted_scores_flat, total_keep, sorted=True)
                global_threshold = final_threshold_tensor[-1]

                for module, name, _ in parameters_to_prune:
                    weight = getattr(module, name)
                    param_name = ptr_to_name[weight.data_ptr()]
                    prev_mask = prev_mask_cache[param_name]
                    score = adjusted_score_cache[param_name]

                    m_local = torch.ge(score, global_threshold).float()
                    curr_mask = m_local.bool()

                    numel = curr_mask.numel()
                    keep_target = curr_mask.sum().item()
                    prev_nonzero_count = prev_mask.sum().item()

                    intersection = (curr_mask & prev_mask).sum().item()
                    union = (curr_mask | prev_mask).sum().item()
                    changed = (curr_mask ^ prev_mask).sum().item()
                    new_regrown = (curr_mask & ~prev_mask).sum().item()
                    pruned = (prev_mask & ~curr_mask).sum().item()

                    total_iou += intersection / (union + 1e-8)
                    total_changed += changed
                    total_new_regrown += new_regrown
                    total_pruned += pruned
                    total_prev_one += prev_nonzero_count
                    total_numel += numel
                    total_keep_target += keep_target

                    layer_count += 1
                    weight.data.mul_(m_local)

            metrics["mask_iou"] = total_iou / layer_count if layer_count > 0 else 0.0
            metrics["mask_flip_rate"] = total_changed / max(total_numel, 1.0)
            metrics["pruned_ratio"] = total_pruned / max(total_prev_one, 1.0)
            metrics["effective_regrowth_ratio"] = total_new_regrown / max(total_keep_target, 1.0)
            metrics["regrowth_ratio"] = metrics["effective_regrowth_ratio"]
            metrics["regrowth_utilization"] = 1.0
            metrics["realized_sparsity"] = 1.0 - (total_keep_target / max(total_numel, 1.0))

            metrics["hysteresis_active"] = 1.0 if hysteresis_active else 0.0
            metrics["hysteresis_threshold"] = float(hysteresis_threshold)
            metrics["hysteresis_band"] = float(hysteresis_band)
            metrics["hysteresis_boundary_fraction"] = float(total_boundary_count / max(total_numel, 1.0))
            metrics["hysteresis_prev_keep_boundary_fraction"] = float(
                total_prev_keep_boundary_count / max(total_boundary_count, 1)
            )

            del initial_net_state
            del global_masks
            del raw_score_cache
            del prev_mask_cache
            del adjusted_score_cache
            del all_raw_scores
            del all_adjusted_scores
            torch.cuda.empty_cache()
        else:
            metrics.update(
                {
                    "mask_iou": 1.0,
                    "mask_flip_rate": 0.0,
                    "pruned_ratio": 0.0,
                    "effective_regrowth_ratio": 0.0,
                    "regrowth_ratio": 0.0,
                    "regrowth_utilization": 0.0,
                    "realized_sparsity": 0.0,
                    "hysteresis_active": 0.0,
                    "hysteresis_threshold": 0.0,
                    "hysteresis_band": 0.0,
                    "hysteresis_boundary_fraction": 0.0,
                    "hysteresis_prev_keep_boundary_fraction": 0.0,
                }
            )

        metrics["sparsity"] = current_sparsity
        metrics["fedmcr_beta"] = current_beta
        return num_samples, metrics

    return train_and_prune


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    config: TestConfig = TestConfig(**_config)
    del _config

    sparse_accuracy = {}
    sparse_loss = {}
    net.to(config.device)
    net.eval()
    avg_exponent = set_spectral_global_exponent(net, False)
    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = net(images)
            per_sample_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    sparse_accuracy["test_accuracy"] = float(correct) / len(cast(Sized, testloader.dataset))
    sparse_loss["loss"] = per_sample_loss / len(cast(Sized, testloader.dataset))
    sparse_accuracy["exponent"] = avg_exponent
    return sparse_loss["loss"], len(cast(Sized, testloader.dataset)), sparse_accuracy


def test_hetero_flash(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    config: TestConfig = TestConfig(**_config)
    del _config

    sparse_accuracy = {}
    sparse_loss = {}
    net.to(config.device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = net(images)
            per_sample_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    sparse_accuracy["test_accuracy"] = float(correct) / len(cast(Sized, testloader.dataset))
    sparse_loss["loss"] = per_sample_loss / len(cast(Sized, testloader.dataset))
    return sparse_loss["loss"], len(cast(Sized, testloader.dataset)), sparse_accuracy


get_train_and_prune = get_fixed_train_and_prune
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
