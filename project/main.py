"""Main entry for prunable-only support-aware aggregation with fail-fast checks.

This version keeps local training/client/dispatch unchanged and only adds
server-side support-aware aggregation. It also validates that prunable flags
are built successfully; if not, it aborts instead of silently running with all-0
support metrics.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import flwr as fl
import hydra
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

from project.client.client import get_client_generator
from project.dispatch.dispatch import dispatch_config, dispatch_data, dispatch_train
from project.fed.server.deterministic_client_manager import DeterministicClientManager
from project.fed.server.wandb_history import WandbHistory
from project.fed.server.wandb_server import WandbServer
from project.fed.utils.utils import (
    get_initial_parameters,
    get_save_parameters_to_file,
    get_weighted_avg_metrics_agg_fn,
    test_client,
)
# IMPORTANT: reuse the task-side helper so server prunable mapping follows the
# same naming path as local training.
from project.task.cifar_resnet18.train_test import get_prunable_param_names_for_server
from project.types.common import ClientGen, FedEvalFN
from project.utils.utils import (
    FileSystemManager,
    RayContextManager,
    seed_everything,
    wandb_init,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["RAY_MEMORY_MONITOR_REFRESH_MS"] = "0"


def _build_prunable_flags_for_sorted_state(net) -> tuple[list[bool], list[str], list[str]]:
    sorted_state_keys = [k for k, _ in sorted(net.state_dict().items(), key=lambda x: x[0])]
    prunable_names = get_prunable_param_names_for_server(net)
    prunable_name_set = set(prunable_names)
    prunable_flags = [key in prunable_name_set for key in sorted_state_keys]
    return prunable_flags, sorted_state_keys, prunable_names


@hydra.main(config_path="conf", config_name="local_cifar_resnet18", version_base=None)
def main(cfg: DictConfig) -> None:
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    original_hydra_dir = Path(hydra.utils.to_absolute_path(HydraConfig.get().runtime.output_dir))
    output_directory = original_hydra_dir if cfg.reuse_output_dir is None else Path(cfg.reuse_output_dir)

    results_dir = output_directory / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.working_dir is not None:
        working_dir = Path(cfg.working_dir)
    else:
        working_dir = output_directory / "working"
    working_dir.mkdir(parents=True, exist_ok=True)

    to_save_once = list(cfg.to_save_once)
    to_clean_once = list(cfg.to_clean_once)
    if "support_agg" not in to_save_once:
        to_save_once.append("support_agg")
    if "support_agg" not in to_clean_once:
        to_clean_once.append("support_agg")

    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,
    ) as _run:
        log(logging.INFO, "Wandb run initialized with %s", cfg.use_wandb)

        with (
            FileSystemManager(
                working_dir=working_dir,
                output_dir=results_dir,
                to_clean_once=to_clean_once,
                to_save_once=to_save_once,
                original_hydra_dir=original_hydra_dir,
                reuse_output_dir=cfg.reuse_output_dir,
                file_limit=cfg.file_limit,
            ) as fs_manager,
            RayContextManager() as _ray_manager,
        ):
            save_files_per_round = fs_manager.get_save_files_every_round(
                cfg.to_save_per_round,
                cfg.save_frequency,
            )
            adjusted_seed = cfg.fed.seed ^ fs_manager.checkpoint_index
            save_parameters_to_file = get_save_parameters_to_file(working_dir)

            client_manager = DeterministicClientManager(adjusted_seed, cfg.fed.enable_resampling)
            history = WandbHistory(cfg.use_wandb)

            net_generator, client_dataloader_gen, fed_dataloater_gen = dispatch_data(cfg)
            train_func, test_func, get_fed_eval_fn = dispatch_train(cfg)
            on_fit_config_fn, on_evaluate_config_fn = dispatch_config(cfg)

            evaluate_fn: FedEvalFN | None = get_fed_eval_fn(
                net_generator,
                fed_dataloater_gen,
                test_func,
                cast(dict, OmegaConf.to_container(cfg.task.fed_test_config)),
                working_dir,
            )

            if cfg.fed.load_saved_parameters:
                parameters_path = results_dir / "parameters" if cfg.fed.use_results_dir else Path(cfg.fed.parameters_folder)
            else:
                parameters_path = None

            initial_parameters = get_initial_parameters(
                net_generator,
                cast(dict, OmegaConf.to_container(cfg.task.net_config_initial_parameters)),
                load_from=parameters_path,
                server_round=cfg.fed.parameters_round,
            )

            strategy = instantiate(
                cfg.strategy.init,
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fed.num_clients_per_round,
                min_evaluate_clients=cfg.fed.num_evaluate_clients_per_round,
                min_available_clients=cfg.fed.num_total_clients,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                evaluate_fn=evaluate_fn,
                accept_failures=False,
                fit_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(cfg.task.fit_metrics),
                evaluate_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(cfg.task.evaluate_metrics),
                initial_parameters=initial_parameters,
            )

            probe_net = net_generator(cast(dict, OmegaConf.to_container(cfg.task.fit_config.net_config)))
            prunable_flags, sorted_state_keys, prunable_names = _build_prunable_flags_for_sorted_state(probe_net)
            num_prunable = int(sum(bool(x) for x in prunable_flags))

            log(logging.INFO, "SupportAgg probe: total sorted state tensors=%d", len(sorted_state_keys))
            log(logging.INFO, "SupportAgg probe: prunable tensors=%d", num_prunable)
            log(logging.INFO, "SupportAgg probe: first 20 prunable names=%s", prunable_names[:20])

            if num_prunable <= 0:
                raise RuntimeError(
                    "SupportAgg prunable mapping failed: num_prunable_tensors == 0. "
                    f"First 20 sorted state keys: {sorted_state_keys[:20]}; "
                    f"Prunable names detected: {prunable_names[:20]}"
                )

            server = WandbServer(
                client_manager=client_manager,
                history=history,
                strategy=strategy,
                save_parameters_to_file=save_parameters_to_file,
                save_files_per_round=save_files_per_round,
                working_dir=working_dir,
                support_cfg=dict(cfg.task.fit_config.run_config),
                prunable_flags=prunable_flags,
                sorted_state_keys=sorted_state_keys,
            )

            client_generator: ClientGen = get_client_generator(
                working_dir=working_dir,
                net_generator=net_generator,
                dataloader_gen=client_dataloader_gen,
                train=train_func,
                test=test_func,
                fed_dataloader_gen=fed_dataloater_gen,
            )

            seed_everything(adjusted_seed)

            test_client(
                test_all_clients=cfg.test_clients.all,
                test_one_client=cfg.test_clients.one,
                client_generator=client_generator,
                initial_parameters=initial_parameters,
                total_clients=cfg.fed.num_total_clients,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
            )

            fl.simulation.start_simulation(
                client_fn=client_generator,
                num_clients=cfg.fed.num_total_clients,
                client_resources={
                    "num_cpus": int(cfg.fed.cpus_per_client),
                    "num_gpus": float(cfg.fed.gpus_per_client),
                },
                server=server,
                config=fl.server.ServerConfig(num_rounds=cfg.fed.num_rounds),
                ray_init_args=(
                    {
                        "num_gpus": 1.0,
                        "include_dashboard": False,
                        "address": cfg.ray_address,
                        "_redis_password": cfg.ray_redis_password,
                        "_node_ip_address": cfg.ray_node_ip_address,
                    }
                    if cfg.ray_address is not None
                    else {"num_gpus": 1.0, "include_dashboard": False}
                ),
                strategy=strategy,
                client_manager=client_manager,
            )

        try:
            completed = subprocess.run(
                ["wandb", "sync", "--clean-old-hours", "24"],
                capture_output=True,
                text=True,
                check=False,
            )
            log(logging.INFO, str(completed))
        except Exception as exc:
            log(logging.INFO, "wandb sync skipped: %s", exc)


if __name__ == "__main__":
    main()
