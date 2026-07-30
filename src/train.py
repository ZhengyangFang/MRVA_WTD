from __future__ import annotations

import copy
import platform
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

from .dataset import (
    TemporalGNNDataset,
    fit_train_scalers,
    load_cache_bundle,
    load_or_build_neighborhood_cache,
    split_sample_table,
)
from .models.temporal_gnn import TemporalGNNModel
from .utils import device_from_config, output_paths, processed_paths, set_seed


class _LogCoshLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return diff + torch.nn.functional.softplus(-2.0 * diff) - torch.log(
            torch.tensor(2.0, device=diff.device, dtype=diff.dtype)
        )


def resolve_experiment(config: dict, experiment_name: str) -> dict:
    cfg = copy.deepcopy(config)
    exp_cfg = cfg.get("experiments", {}).get(experiment_name, {})
    if "model_type" in exp_cfg:
        cfg["model"]["model_type"] = exp_cfg["model_type"]
    if "graph_type" in exp_cfg:
        cfg["graph"]["graph_type"] = exp_cfg["graph_type"]
    return cfg


def _load_graph_for_experiment(config: dict, experiment_name: str) -> dict[str, Any] | None:
    graph_type = config["graph"].get("graph_type")
    if graph_type is None:
        return None
    graph_path = output_paths(config, experiment_name)["graph_dir"] / f"{graph_type}_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}. Run graph building first.")
    return torch.load(graph_path, map_location="cpu", weights_only=False)


def _loader_runtime_kwargs(config: dict, device: str) -> dict[str, Any]:
    requested_num_workers = int(config["training"].get("num_workers", 0))
    num_workers = requested_num_workers
    use_cuda = str(device).startswith("cuda")
    if platform.system().lower() == "windows" and requested_num_workers > 0:
        warnings.warn(
            "Falling back to num_workers=0 on Windows because multi-worker PyG loading "
            "can hang in this environment. pin_memory/AMP/cache acceleration remain enabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        num_workers = 0
    kwargs: dict[str, Any] = {"num_workers": num_workers}
    pin_memory = bool(config["training"].get("pin_memory", use_cuda)) and use_cuda
    if num_workers > 0:
        kwargs["pin_memory"] = pin_memory
        kwargs["persistent_workers"] = bool(config["training"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
    else:
        kwargs["pin_memory"] = pin_memory
    return kwargs


def _amp_enabled(config: dict, device: str) -> bool:
    return bool(config["training"].get("use_amp", True)) and str(device).startswith("cuda")


def _build_dataloaders(
    config: dict,
    experiment_name: str,
    scalers: dict[str, Any],
    bundle=None,
):
    bundle = bundle or load_cache_bundle(config)
    splits = split_sample_table(bundle.sample_table)
    batch_size = int(config["training"]["batch_size"])
    device = device_from_config()
    loader_kwargs = _loader_runtime_kwargs(config, device)

    graph_bundle = _load_graph_for_experiment(config, experiment_name)
    graph_path = output_paths(config, experiment_name)["graph_dir"] / f"{config['graph']['graph_type']}_graph.pt"
    neighborhood_cache = load_or_build_neighborhood_cache(
        config,
        bundle.sample_table,
        graph_bundle,
        graph_path,
        int(config["graph"]["graph_hops"]),
    )
    datasets = {
        split: TemporalGNNDataset(
            config,
            frame,
            bundle.dynamic_cache,
            bundle.aem_cache,
            bundle.topo_cache,
            graph_bundle,
            scalers,
            neighborhood_cache,
        )
        for split, frame in splits.items()
    }
    loaders = {
        split: PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            **loader_kwargs,
        )
        for split, dataset in datasets.items()
    }
    return bundle, datasets, loaders


def _build_model(config: dict, bundle, scalers: dict[str, Any]):
    aem_profile_dim = bundle.aem_cache["profiles"].shape[1]
    topo_dim = bundle.topo_cache["features"].shape[1]
    return TemporalGNNModel(aem_profile_dim, topo_dim, config)


def _initialize_model_from_checkpoint(model: nn.Module, config: dict, experiment_name: str) -> Path | None:
    init_path_value = str(config.get("training", {}).get("init_checkpoint_path", "")).strip()
    if not init_path_value:
        return None

    raw_path = Path(init_path_value)
    candidates = [raw_path]
    config_path = config.get("__config_path__")
    if config_path:
        repo_root = Path(config_path).resolve().parent.parent
        candidates.append(repo_root / raw_path)

    checkpoint_path = next((path for path in candidates if path.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Initialization checkpoint not found for {experiment_name}: {init_path_value}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")
    strict = bool(config.get("training", {}).get("init_checkpoint_strict", True))
    model.load_state_dict(state_dict, strict=strict)
    print(f"Initialized {experiment_name} from checkpoint: {checkpoint_path}")
    return checkpoint_path



def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch[key]
    return getattr(batch, key)


def _batch_vector(batch: Any, key: str, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    if isinstance(batch, dict):
        if key not in batch:
            return None
        value = batch[key]
    else:
        if not hasattr(batch, key):
            return None
        value = getattr(batch, key)
    return torch.as_tensor(value, dtype=dtype, device=device).view(-1)


def _step_weights(values: torch.Tensor, thresholds: list[float], weights: list[float]) -> torch.Tensor:
    if not thresholds or not weights:
        return torch.ones_like(values, dtype=torch.float32)
    if len(weights) != len(thresholds) + 1:
        raise ValueError(
            "Weight configuration is invalid: weights length must equal thresholds length + 1."
        )
    out = torch.full_like(values, float(weights[0]), dtype=torch.float32)
    for threshold, weight in zip(thresholds, weights[1:]):
        out = torch.where(values >= float(threshold), torch.full_like(out, float(weight)), out)
    return out


def _sample_loss_weights(
    batch: Any,
    config: dict[str, Any],
    device: str,
    split_name: str | None = None,
) -> torch.Tensor | None:
    weighting_cfg = config.get("training", {}).get("loss_weighting", {})
    if not bool(weighting_cfg.get("enabled", False)):
        return None

    h_steps = _batch_vector(batch, "h_steps", device, torch.long)
    delta_h_true = _batch_vector(batch, "delta_h_true", device, torch.float32)
    if h_steps is None or delta_h_true is None:
        return None

    weights = torch.ones_like(delta_h_true, dtype=torch.float32)

    horizon_weights = weighting_cfg.get("horizon_weights", {})
    for horizon, weight in horizon_weights.items():
        weights = torch.where(
            h_steps == int(horizon),
            weights * float(weight),
            weights,
        )

    quality_cfg = copy.deepcopy(weighting_cfg.get("quality", {}))
    if split_name:
        split_overrides = quality_cfg.get("split_overrides", {})
        override_cfg = split_overrides.get(str(split_name), {})
        for key, value in override_cfg.items():
            quality_cfg[key] = value
    if bool(quality_cfg.get("enabled", False)):
        n_days_t = _batch_vector(batch, "n_days_observed_t", device, torch.float32)
        n_days_target = _batch_vector(batch, "n_days_observed_target", device, torch.float32)
        if n_days_t is not None and n_days_target is not None:
            min_days = torch.minimum(n_days_t, n_days_target)
            day_thresholds = [float(v) for v in quality_cfg.get("min_observation_day_thresholds", [])]
            day_weights = [float(v) for v in quality_cfg.get("min_observation_day_weights", [])]
            weights = weights * _step_weights(min_days, day_thresholds, day_weights)

        n_sites_t = _batch_vector(batch, "n_unique_sites_t", device, torch.float32)
        n_sites_target = _batch_vector(batch, "n_unique_sites_target", device, torch.float32)
        if n_sites_t is not None and n_sites_target is not None:
            min_sites = torch.minimum(n_sites_t, n_sites_target)
            site_thresholds = [float(v) for v in quality_cfg.get("min_unique_site_thresholds", [])]
            site_weights = [float(v) for v in quality_cfg.get("min_unique_site_weights", [])]
            weights = weights * _step_weights(min_sites, site_thresholds, site_weights)

    if bool(weighting_cfg.get("normalize", True)):
        weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def _aux_loss_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("training", {}).get("aux_loss", {}))


def _correlation_std_aux_loss(pred: torch.Tensor, target: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    aux_cfg = _aux_loss_cfg(config)
    if not bool(aux_cfg.get("enabled", False)):
        return pred.new_tensor(0.0)

    pred_f = pred.view(-1).float()
    target_f = target.view(-1).float()
    if pred_f.numel() == 0:
        return pred.new_tensor(0.0)

    corr_weight = float(aux_cfg.get("corr_weight", 0.0))
    std_weight = float(aux_cfg.get("std_weight", 0.0))
    eps = float(aux_cfg.get("eps", 1e-6))

    total = pred.new_tensor(0.0, dtype=torch.float32)

    if corr_weight > 0.0 and pred_f.numel() >= 2:
        pred_centered = pred_f - pred_f.mean()
        target_centered = target_f - target_f.mean()
        denom = torch.sqrt(
            torch.sum(pred_centered * pred_centered) * torch.sum(target_centered * target_centered)
        ).clamp_min(eps)
        corr = torch.sum(pred_centered * target_centered) / denom
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        total = total + corr_weight * (1.0 - corr)

    if std_weight > 0.0:
        pred_std = torch.std(pred_f, unbiased=False)
        target_std = torch.std(target_f, unbiased=False)
        total = total + std_weight * torch.abs(pred_std - target_std)

    return total.to(dtype=pred.dtype)


def _forward_loss(
    model,
    batch,
    config: dict,
    device: str,
    loss_fn: nn.Module,
    amp_enabled: bool = False,
    split_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
        batch = batch.to(device)
        model_out = model(batch)
        target = batch.y.view(-1)
        pred = model_out
        loss_per_sample = loss_fn(pred, target).view(-1)
        sample_weights = _sample_loss_weights(batch, config, device, split_name=split_name)
        if sample_weights is None:
            loss = loss_per_sample.mean()
        else:
            loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        loss = loss + _correlation_std_aux_loss(pred, target, config)
    return pred, loss


def _evaluate_loss(
    model,
    loader,
    config: dict,
    device: str,
    loss_fn: nn.Module,
    amp_enabled: bool = False,
    split_name: str | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            pred, loss = _forward_loss(
                model,
                batch,
                config,
                device,
                loss_fn,
                amp_enabled=amp_enabled,
                split_name=split_name,
            )
            n = pred.shape[0]
            total_loss += float(loss.item()) * n
            total_count += n
    return total_loss / max(total_count, 1)


def _loss_from_config(config: dict) -> nn.Module:
    loss_name = str(config["training"]["loss"]).lower()
    if loss_name == "mse":
        return nn.MSELoss(reduction="none")
    if loss_name == "mae":
        return nn.L1Loss(reduction="none")
    if loss_name == "smooth_l1":
        beta = float(config["training"].get("smooth_l1_beta", 1.0))
        return nn.SmoothL1Loss(beta=beta, reduction="none")
    if loss_name == "log_cosh":
        return _LogCoshLoss()
    huber_delta = float(config["training"].get("huber_delta", 1.0))
    return nn.HuberLoss(delta=huber_delta, reduction="none")


def train_experiment(config: dict, experiment_name: str) -> Path:
    config = resolve_experiment(config, experiment_name)
    set_seed(int(config["training"]["seed"]))
    paths = output_paths(config, experiment_name)
    processed = processed_paths(config)

    bundle = load_cache_bundle(config)
    scalers_path = Path(processed["scalers_pt"])
    if scalers_path.exists():
        scalers = torch.load(scalers_path, map_location="cpu", weights_only=False)
    else:
        scalers = fit_train_scalers(config, bundle.sample_table, bundle.dynamic_cache, bundle.aem_cache, bundle.topo_cache)
        torch.save(scalers, scalers_path)

    device = device_from_config()
    amp_enabled = _amp_enabled(config, device)
    bundle, datasets, loaders = _build_dataloaders(config, experiment_name, scalers, bundle=bundle)
    model = _build_model(config, bundle, scalers).to(device)
    init_checkpoint_path = _initialize_model_from_checkpoint(model, config, experiment_name)
    loss_fn = _loss_from_config(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    patience = int(config["training"]["patience"])
    wait = 0
    history = []
    checkpoint_path = paths["experiment_dir"] / "best_checkpoint.pt"
    max_epochs = int(config["training"]["max_epochs"])
    progress = tqdm(
        range(1, max_epochs + 1),
        total=max_epochs,
        desc=f"Training {experiment_name}",
        leave=True,
    )

    for epoch in progress:
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            pred, loss = _forward_loss(
                model,
                batch,
                config,
                device,
                loss_fn,
                amp_enabled=amp_enabled,
                split_name="train",
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = pred.shape[0]
            running_loss += float(loss.item()) * n
            running_count += n

        train_loss = running_loss / max(running_count, 1)
        val_loss = _evaluate_loss(
            model,
            loaders["validation"],
            config,
            device,
            loss_fn,
            amp_enabled=amp_enabled,
            split_name="validation",
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(
                {
                    "experiment_name": experiment_name,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "scalers": scalers,
                    "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else "",
                    "best_val_loss": best_val,
                    "history": history,
                },
                checkpoint_path,
            )
        else:
            wait += 1
            if wait >= patience:
                progress.set_postfix(
                    epoch=f"{epoch}/{max_epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    best_val=f"{best_val:.4f}",
                    early_stop="yes",
                )
                break
        progress.set_postfix(
            epoch=f"{epoch}/{max_epochs}",
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            best_val=f"{best_val:.4f}",
        )

    progress.close()
    pd.DataFrame(history).to_csv(paths["experiment_dir"] / "train_history.csv", index=False, lineterminator="\n")
    return checkpoint_path


def train_all_experiments(config: dict, experiments: list[str] | None = None) -> dict[str, Path]:
    experiments = experiments or list(config["experiments"].keys())
    out = {}
    for experiment_name in experiments:
        out[experiment_name] = train_experiment(config, experiment_name)
    return out
