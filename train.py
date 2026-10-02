#!/usr/bin/env python3
"""
Train Custom CNN and ResNet-from-scratch on Brain Tumor MRI dataset.
Optionally trains transfer-learning models when torchvision is available.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders, CLASS_NAMES
from src.models import get_model
from src.utils import (
    train_one_epoch,
    evaluate,
    compute_metrics,
    plot_confusion_matrix,
    plot_history,
    save_checkpoint,
)


def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 224,
    lr: float = 1e-3,
    device: str = "cpu",
    pretrained: bool = False,
):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}  |  device={device}  |  epochs={epochs}")
    print(f"{'='*60}")

    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size, num_workers=0
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    print(f"Test  samples: {len(test_loader.dataset)}")

    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = Path(output_dir) / f"{model_name}_best.pth"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": val_acc, "val_loss": val_loss},
                str(best_path),
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    # Final evaluation on test set with best weights
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["test_loss"] = float(test_loss)
    metrics["test_acc"] = float(test_acc)

    print("\n--- Test Set Results ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-macro : {metrics['f1_macro']:.4f}")
    print(metrics["report"])

    # Save artifacts
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_history(history, str(out / f"{model_name}_history.png"))
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(out / f"{model_name}_cm.png"),
        title=f"{model_name} – Confusion Matrix",
    )
    with open(out / f"{model_name}_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"}, f, indent=2)
        f.write("\n\nClassification Report:\n")
        f.write(metrics["report"])

    with open(out / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)

    print(f"\nArtifacts saved to {out}")
    return metrics, str(best_path)


def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI classifiers")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--output-dir", default="models", help="Where to save weights")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["custom_cnn", "resnet_scratch"],
        help="Models to train (custom_cnn resnet_scratch mobilenet_v2 resnet50 ...)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    for name in args.models:
        try:
            pretrained = name not in ("custom_cnn", "resnet_scratch")
            metrics, path = train_model(
                model_name=name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                lr=args.lr,
                device=device,
                pretrained=pretrained,
            )
            results[name] = {"metrics": metrics, "path": path}
        except Exception as e:
            print(f"[ERROR] Failed to train {name}: {e}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Accuracy)")
    print("=" * 60)
    for name, res in results.items():
        acc = res["metrics"].get("accuracy", 0)
        f1 = res["metrics"].get("f1_macro", 0)
        print(f"  {name:20s}  Acc={acc:.4f}  F1-macro={f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
