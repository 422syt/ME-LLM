#!/usr/bin/env python3
"""Train, select, and evaluate ME-LLM with validation-only model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from torch import nn, optim
from tqdm.auto import tqdm

from data_provider.data_factory import data_provider
from models import MELLM


DATASET_DIMENSIONS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Weather": 21,
    "Traffic": 862,
}
LEARNING_RATES = (5e-4, 1e-3, 2e-3)
FORECAST_HORIZONS = (96, 192, 336, 720)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ME-LLM forecasting")
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--task_name", default="long_term_forecast")
    parser.add_argument("--model", default="MELLM", choices=("MELLM",))
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--data", required=True, choices=tuple(DATASET_DIMENSIONS))
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M", choices=("M", "S", "MS"))
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", required=True)
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--seasonal_patterns", default="Monthly")

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, required=True, choices=FORECAST_HORIZONS)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--dec_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, required=True)

    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_prototypes", type=int, default=1000)
    parser.add_argument("--prototype_rank", type=int, default=96)
    parser.add_argument("--semantic_heads", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_prompt_tokens", type=int, default=176)
    parser.add_argument("--max_prediction_length", type=int, default=720)
    parser.add_argument("--prototype_attention_chunk_size", type=int, default=64)
    parser.add_argument("--parameter_count_tolerance", type=int, default=50000)

    parser.add_argument("--llm_model", default="BERT", choices=("BERT",))
    parser.add_argument("--llm_layers", type=int, default=12)
    parser.add_argument("--bert_path", required=True)
    parser.add_argument(
        "--local_files_only",
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=True,
    )
    parser.add_argument("--output_attention", action="store_true")

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, required=True, choices=LEARNING_RATES)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)

    parser.add_argument("--run_root", default="./runs")
    parser.add_argument("--run_name")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--deterministic_algorithms", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    expected_dim = DATASET_DIMENSIONS[args.data]
    dimensions = (args.enc_in, args.dec_in, args.c_out)
    if dimensions != (expected_dim, expected_dim, expected_dim):
        raise ValueError(
            f"{args.data} requires enc_in=dec_in=c_out={expected_dim}; received {dimensions}."
        )
    if args.seq_len != 512:
        raise ValueError("The controlled protocol uses seq_len=512.")
    if args.patch_len != 16 or args.stride != 8:
        raise ValueError("The controlled protocol uses patch_len=16 and stride=8.")
    if args.batch_size != 24:
        raise ValueError("The controlled protocol uses batch_size=24.")
    if args.train_epochs != 15 or args.patience != 5:
        raise ValueError("The controlled protocol uses 15 epochs and patience 5.")
    if args.llm_layers != 12:
        raise ValueError("The controlled protocol uses all 12 BERT-base encoder layers.")
    if args.mode == "test" and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required in test mode.")


def set_seed(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic_algorithms
    torch.backends.cudnn.benchmark = not deterministic_algorithms
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "accelerate": accelerate.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_identifier(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    learning_rate = f"{args.learning_rate:.4g}".replace(".", "p")
    return (
        f"{args.model}_{args.data}_H{args.pred_len}_seed{args.seed}_lr{learning_rate}"
    )


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def prepare_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    decoder_zeros = torch.zeros_like(batch_y[:, -args.pred_len :, :])
    decoder_input = torch.cat(
        [batch_y[:, : args.label_len, :], decoder_zeros], dim=1
    )
    target = batch_y[:, -args.pred_len :, :]
    return batch_x, target, batch_x_mark, batch_y_mark, decoder_input


def model_output(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_x_mark: torch.Tensor,
    decoder_input: torch.Tensor,
    batch_y_mark: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    output = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
    feature_start = -1 if args.features == "MS" else 0
    return output[:, -args.pred_len :, feature_start:]


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_squared_error = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    progress = tqdm(
        loader,
        disable=not accelerator.is_local_main_process,
        desc="train",
        leave=False,
    )
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_x, target, batch_x_mark, batch_y_mark, decoder_input = prepare_batch(
            batch, args, accelerator.device
        )
        with accelerator.autocast():
            prediction = model_output(
                model,
                batch_x,
                batch_x_mark,
                decoder_input,
                batch_y_mark,
                args,
            )
            loss = nn.functional.mse_loss(prediction, target)
        accelerator.backward(loss)
        if args.gradient_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()

        difference = (prediction.detach() - target).double()
        total_squared_error += difference.square().sum()
        total_count += difference.numel()
        progress.set_postfix(mse=f"{loss.item():.6f}")

    totals = accelerator.reduce(
        torch.stack((total_squared_error, total_count)), reduction="sum"
    )
    return float((totals[0] / totals[1]).item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    model.eval()
    total_squared_error = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_absolute_error = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    for batch in tqdm(
        loader,
        disable=not accelerator.is_local_main_process,
        desc="evaluate",
        leave=False,
    ):
        batch_x, target, batch_x_mark, batch_y_mark, decoder_input = prepare_batch(
            batch, args, accelerator.device
        )
        with accelerator.autocast():
            prediction = model_output(
                model,
                batch_x,
                batch_x_mark,
                decoder_input,
                batch_y_mark,
                args,
            )
        difference = (prediction - target).double()
        total_squared_error += difference.square().sum()
        total_absolute_error += difference.abs().sum()
        total_count += difference.numel()

    totals = accelerator.reduce(
        torch.stack((total_squared_error, total_absolute_error, total_count)),
        reduction="sum",
    )
    mse = float((totals[0] / totals[2]).item())
    mae = float((totals[1] / totals[2]).item())
    return mse, mae


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {name: state[name].detach().cpu() for name in sorted(trainable_names)}


def save_checkpoint(
    model: nn.Module,
    accelerator: Accelerator,
    checkpoint_path: Path,
    args: argparse.Namespace,
    epoch: int,
    validation_mse: float,
    validation_mae: float,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "format_version": 1,
                "trainable_state_dict": trainable_state_dict(unwrapped),
                "epoch": epoch,
                "validation_mse": validation_mse,
                "validation_mae": validation_mae,
                "configuration": vars(args),
            },
            checkpoint_path,
        )
    accelerator.wait_for_everyone()


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_state = checkpoint["trainable_state_dict"]
    parameters = dict(model.named_parameters())
    expected_names = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    saved_names = set(saved_state)
    if saved_names != expected_names:
        missing = sorted(expected_names - saved_names)
        extra = sorted(saved_names - expected_names)
        raise ValueError(f"Checkpoint parameter mismatch; missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, tensor in saved_state.items():
            parameters[name].copy_(tensor)
    return checkpoint


def validate_checkpoint_configuration(
    checkpoint: Dict[str, Any], args: argparse.Namespace
) -> None:
    saved = checkpoint.get("configuration")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint configuration is missing.")
    fields = (
        "model",
        "data",
        "pred_len",
        "seed",
        "seq_len",
        "patch_len",
        "stride",
        "num_prototypes",
        "prototype_rank",
        "semantic_heads",
        "llm_layers",
        "enc_in",
        "learning_rate",
        "bert_path",
    )
    mismatches = [
        f"{field}: checkpoint={saved.get(field)!r}, run={getattr(args, field)!r}"
        for field in fields
        if saved.get(field) != getattr(args, field)
    ]
    if mismatches:
        raise ValueError("Checkpoint configuration mismatch: " + "; ".join(mismatches))


def build_model(args: argparse.Namespace) -> nn.Module:
    model = MELLM.Model(args).float()
    report = model.parameter_report()
    if report["trainable"] <= 0:
        raise ValueError("The model has no trainable parameters.")
    return model


def run_train(args: argparse.Namespace, accelerator: Accelerator) -> None:
    run_dir = Path(args.run_root) / run_identifier(args)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    validation_path = run_dir / "validation_metrics.json"

    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(vars(args), config_path)
    accelerator.wait_for_everyone()

    train_data, train_loader = data_provider(args, "train")
    validation_data, validation_loader = data_provider(args, "val")
    del train_data, validation_data

    model = build_model(args)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, train_loader, validation_loader
    )

    best_validation_mse = float("inf")
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.train_epochs + 1):
        epoch_start = time.time()
        train_mse = train_epoch(model, train_loader, optimizer, accelerator, args)
        validation_mse, validation_mae = evaluate(
            model, validation_loader, accelerator, args
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_mae": validation_mae,
                "seconds": time.time() - epoch_start,
            }
        )
        accelerator.print(
            f"epoch={epoch:02d} train_mse={train_mse:.7f} "
            f"validation_mse={validation_mse:.7f} "
            f"validation_mae={validation_mae:.7f}"
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_validation_mae = validation_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                accelerator,
                checkpoint_path,
                args,
                epoch,
                validation_mse,
                validation_mae,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            accelerator.print(f"early_stopping_epoch={epoch}")
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "record_type": "validation_selection",
            "model": args.model,
            "dataset": args.data,
            "horizon": args.pred_len,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_mse,
            "best_validation_mae": best_validation_mae,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "config_file": str(config_path.resolve()),
            "history": history,
            "elapsed_seconds": time.time() - start_time,
            "git_commit": git_commit(),
            "environment": runtime_environment(),
        }
        atomic_json_dump(payload, validation_path)
        accelerator.print(f"validation_record={validation_path}")


def run_test(args: argparse.Namespace, accelerator: Accelerator) -> None:
    run_dir = Path(args.run_root) / run_identifier(args)
    config_path = run_dir / "test_config.json"
    result_path = run_dir / "test_metrics.json"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(vars(args), config_path)
    accelerator.wait_for_everyone()

    test_data, test_loader = data_provider(args, "test")
    del test_data
    model = build_model(args)
    checkpoint = load_checkpoint(model, checkpoint_path)
    validate_checkpoint_configuration(checkpoint, args)
    model, test_loader = accelerator.prepare(model, test_loader)

    start_time = time.time()
    test_mse, test_mae = evaluate(model, test_loader, accelerator, args)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "record_type": "test_evaluation",
            "model": args.model,
            "dataset": args.data,
            "horizon": args.pred_len,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "mse": test_mse,
            "mae": test_mae,
            "selected_epoch": checkpoint["epoch"],
            "selected_validation_mse": checkpoint["validation_mse"],
            "selected_validation_mae": checkpoint["validation_mae"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "config_file": str(config_path.resolve()),
            "elapsed_seconds": time.time() - start_time,
            "git_commit": git_commit(),
            "environment": runtime_environment(),
        }
        atomic_json_dump(payload, result_path)
        accelerator.print(
            f"test_mse={test_mse:.7f} test_mae={test_mae:.7f} "
            f"test_record={result_path}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed, args.deterministic_algorithms)
    accelerator = Accelerator()
    if args.mode == "train":
        run_train(args, accelerator)
    else:
        run_test(args, accelerator)


if __name__ == "__main__":
    main()
