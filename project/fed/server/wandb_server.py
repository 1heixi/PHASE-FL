"""Flower server accounting for Weights&Biases+file saving.

Prunable-only Soft Support-Aware Aggregation:
- keep client/local training unchanged
- keep Strategy object as FedAvg (or whatever cfg.strategy.init instantiates)
- inject support-aware soft aggregation only on *prunable* weights after the
  original strategy.aggregate_fit
- export per-prunable-tensor debiased layer confidence for the local third innovation
"""

from __future__ import annotations

import time
import timeit
from collections.abc import Callable
from logging import INFO
from pathlib import Path
from typing import Any

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

from project.fed.utils.support_aware_aggregation_utils import (
    support_aware_aggregate_from_results_prunable_only,
)


class WandbServer(Server):
    """Flower server with optional server-side support-aware aggregation on prunable weights."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Strategy | None = None,
        history: History | None = None,
        save_parameters_to_file: Callable[[Parameters], None],
        save_files_per_round: Callable[[int], None],
        working_dir: str | Path | None = None,
        support_cfg: dict[str, Any] | None = None,
        prunable_flags: list[bool] | None = None,
        sorted_state_keys: list[str] | None = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)

        self.history: History | None = history
        self.save_parameters_to_file = save_parameters_to_file
        self.save_files_per_round = save_files_per_round
        self.support_working_dir = Path(working_dir or ".")

        cfg = support_cfg or {}
        self.support_agg_activate_round = int(cfg.get("support_agg_activate_round", 2000))
        self.support_ema_beta = float(cfg.get("support_ema_beta", 0.90))
        self.support_gamma = float(cfg.get("support_gamma", 2.0))
        self.support_alpha_min = float(cfg.get("support_alpha_min", 0.05))
        self.support_update_interval = int(cfg.get("support_update_interval", 1))
        self.prunable_flags = list(prunable_flags or [])
        self.sorted_state_keys = list(sorted_state_keys or [])

        if self.strategy is not None:
            original_aggregate_fit = self.strategy.aggregate_fit

            def wrapped_aggregate_fit(server_round, results, failures):
                start_time = time.time()
                agg_res = original_aggregate_fit(server_round, results, failures)
                agg_time = time.time() - start_time

                if agg_res is None:
                    return agg_res

                parameters_aggregated, metrics_aggregated = agg_res
                if metrics_aggregated is None:
                    metrics_aggregated = {}

                if self.parameters is not None and results and self.prunable_flags:
                    prev_global_arrays = parameters_to_ndarrays(self.parameters)
                    base_aggregated_arrays = parameters_to_ndarrays(parameters_aggregated)

                    updated_arrays, support_metrics = support_aware_aggregate_from_results_prunable_only(
                        server_round=server_round,
                        prev_global_arrays=prev_global_arrays,
                        baseline_aggregated_arrays=base_aggregated_arrays,
                        results=results,
                        working_dir=self.support_working_dir,
                        support_agg_activate_round=self.support_agg_activate_round,
                        support_ema_beta=self.support_ema_beta,
                        support_gamma=self.support_gamma,
                        support_alpha_min=self.support_alpha_min,
                        support_update_interval=self.support_update_interval,
                        prunable_flags=self.prunable_flags,
                        sorted_state_keys=self.sorted_state_keys,
                    )
                    metrics_aggregated.update(support_metrics)
                    if updated_arrays is not None:
                        parameters_aggregated = ndarrays_to_parameters(updated_arrays)
                else:
                    metrics_aggregated.update(
                        {
                            "support_agg_active": 0.0,
                            "support_mean": 0.0,
                            "support_ema_mean": 0.0,
                            "support_alpha_mean": 0.0,
                            "support_nonzero_fraction": 0.0,
                            "support_prunable_fraction": float(sum(self.prunable_flags) / len(self.prunable_flags))
                            if self.prunable_flags
                            else 0.0,
                            "support_num_prunable_tensors": float(sum(self.prunable_flags)) if self.prunable_flags else 0.0,
                        }
                    )

                metrics_aggregated["server_aggregation_time"] = float(agg_time)
                return parameters_aggregated, metrics_aggregated

            self.strategy.aggregate_fit = wrapped_aggregate_fit

    def fit(self, num_rounds: int, timeout: float | None) -> History:
        history = self.history if self.history is not None else History()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        self.save_parameters_to_file(self.parameters)
        self.save_files_per_round(0)

        for current_round in range(1, num_rounds + 1):
            timestamp = time.time()
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            fit_round_time = time.time() - timestamp

            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                metrics_cen["fit_round_time"] = fit_round_time
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

            self.save_parameters_to_file(self.parameters)
            self.save_files_per_round(current_round)

        elapsed = timeit.default_timer() - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
