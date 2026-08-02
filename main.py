import argparse
import json
import time
from pathlib import Path

import torch

from config import get_config
from dataset import get_dataloaders
from model import MODEL_NAME, get_model
from trainer import Trainer, evaluate, plot_training_history, set_seed

SUPPORTED_FEATURES = ["mel", "fbank", "pcen"]


def run_experiment(
    feature_type: str,
    data_root: str,
    output_dir: str,
    config_overrides: dict | None = None,
) -> dict:
    config = get_config(feature_type)
    config.data_root = data_root
    config.output_dir = output_dir
    config.exp_name = f"{feature_type}_{MODEL_NAME}"

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)

    experiment_dir = Path(output_dir) / f"{feature_type}_{MODEL_NAME}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.train.seed)

    print(f"Feature: {feature_type}")
    print(f"Model: {MODEL_NAME}")
    print("Loading data...")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(config)
    model = get_model(config)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        log_dir=str(experiment_dir),
    )

    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time

    plot_training_history(
        history,
        str(experiment_dir / "training_history.png"),
    )

    device = torch.device(
        config.train.device if torch.cuda.is_available() else "cpu"
    )
    test_results = evaluate(
        model=trainer.model,
        test_loader=test_loader,
        device=device,
        save_path=str(experiment_dir / "test_results.json"),
    )

    checkpoint_path = experiment_dir / "best_model.pt"
    trainer.save_checkpoint(str(checkpoint_path))

    configuration = {
        "feature_type": feature_type,
        "model_name": MODEL_NAME,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "dropout": config.model.dropout,
        "num_epochs": config.train.num_epochs,
        "use_augmentation": config.train.use_augmentation,
        "mixup_alpha": config.train.mixup_alpha,
        "spec_augment": config.train.spec_augment,
        "seed": config.train.seed,
    }

    with open(experiment_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    summary = {
        "feature_type": feature_type,
        "model_name": MODEL_NAME,
        "training_time_seconds": training_time,
        "total_parameters": total_parameters,
        "best_validation_f1": trainer.best_val_f1,
        "test_results": test_results,
        "checkpoint": str(checkpoint_path),
    }

    with open(experiment_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Training time: {training_time / 60:.2f} minutes")
    print(f"Checkpoint: {checkpoint_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CBSkyNet.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--feature",
        type=str,
        default="pcen",
        choices=SUPPORTED_FEATURES,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    config_overrides = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "use_augmentation": not args.no_augment,
        "seed": args.seed,
    }

    run_experiment(
        feature_type=args.feature,
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_overrides=config_overrides,
    )

if __name__ == "__main__":
    main()
