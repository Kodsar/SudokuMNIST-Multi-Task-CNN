import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_datasets
from model import SudokuNet


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Select CUDA, Apple MPS, or CPU automatically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device, ce_missing, ce_sorted, reg_loss, w_sorted):
    """
    Evaluation loop for train/val/test sets.
    Returns: loss, missing acc, sort acc, sum MAE
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    correct_missing = 0
    total_missing = 0

    correct_sorted = 0
    total_sorted = 0

    mae_sum_total = 0.0
    sum_count = 0

    for batch in loader:
        x = batch["image"].to(device)
        y_missing = batch["missing_digit"].to(device)
        y_sorted = batch["sorted_labels"].to(device)  # [B,6]
        y_sum = batch["sum_labels"].to(device)        # [B,6]

        out = model(x)

        loss_missing = ce_missing(out["missing_logits"], y_missing)
        loss_sorted = ce_sorted(out["sorted_logits"].reshape(-1, 3), y_sorted.reshape(-1))
        loss_sum = reg_loss(out["sum_pred"], y_sum)

        loss = loss_missing + w_sorted * loss_sorted + loss_sum

        total_loss += loss.item()
        n_batches += 1

        correct_missing += (out["missing_logits"].argmax(1) == y_missing).sum().item()
        total_missing += y_missing.numel()

        correct_sorted += (out["sorted_logits"].argmax(2) == y_sorted).sum().item()
        total_sorted += y_sorted.numel()

        mae_sum_total += torch.abs(out["sum_pred"] - y_sum).sum().item()
        sum_count += y_sum.numel()

    return {
        "loss": total_loss / max(1, n_batches),
        "acc_missing": correct_missing / max(1, total_missing),
        "acc_sorted": correct_sorted / max(1, total_sorted),
        "mae_sum": mae_sum_total / max(1, sum_count),
    }


def save_plots(history, test_metrics, out_dir):
    """
    Save required plots to out_dir.
    Produces:
      - loss.png
      - missing_digit.png
      - sort_order.png
      - sum_prediction.png
      - training_history.png (2x2 like the sample)
    """
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
     

    # 1) Total loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o")
    plt.plot(epochs, history["val_loss"], marker="s")
    plt.axhline(test_metrics["loss"], linestyle="--")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Total Loss", "Val Total Loss", f"Test Loss: {test_metrics['loss']:.4f}"])
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Missing digit task: plot loss + acc (two y concepts, but single axis like sample is ok)
    plt.figure()
    plt.plot(epochs, history["train_loss_missing"], marker="o")
    plt.plot(epochs, history["val_loss_missing"], marker="s")
    plt.plot(epochs, history["train_acc_missing"], marker="^")
    plt.plot(epochs, history["val_acc_missing"], marker="D")
    plt.axhline(test_metrics["loss_missing"], linestyle="--")
    plt.axhline(test_metrics["acc_missing"], linestyle="--")
    plt.title("Missing Digit Task")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend([
        "Train Loss", "Val Loss", "Train Acc", "Val Acc",
        f"Test Loss: {test_metrics['loss_missing']:.4f}",
        f"Test Acc: {test_metrics['acc_missing']:.4f}",
    ])
    plt.savefig(os.path.join(out_dir, "missing_digit.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Sort order task
    plt.figure()
    plt.plot(epochs, history["train_loss_sorted"], marker="o")
    plt.plot(epochs, history["val_loss_sorted"], marker="s")
    plt.plot(epochs, history["train_acc_sorted"], marker="^")
    plt.plot(epochs, history["val_acc_sorted"], marker="D")
    plt.axhline(test_metrics["loss_sorted"], linestyle="--")
    plt.axhline(test_metrics["acc_sorted"], linestyle="--")
    plt.title("Sort Order Task")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend([
        "Train Loss", "Val Loss", "Train Acc", "Val Acc",
        f"Test Loss: {test_metrics['loss_sorted']:.4f}",
        f"Test Acc: {test_metrics['acc_sorted']:.4f}",
    ])
    plt.savefig(os.path.join(out_dir, "sort_order.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Sum prediction task (loss + MAE)
    plt.figure()
    plt.plot(epochs, history["train_loss_sum"], marker="o")
    plt.plot(epochs, history["val_loss_sum"], marker="s")
    plt.plot(epochs, history["train_mae_sum"], marker="^")
    plt.plot(epochs, history["val_mae_sum"], marker="D")
    plt.axhline(test_metrics["loss_sum"], linestyle="--")
    plt.axhline(test_metrics["mae_sum"], linestyle="--")
    plt.title("Sum Prediction Task")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.legend([
        "Train Loss", "Val Loss", "Train MAE", "Val MAE",
        f"Test Loss: {test_metrics['loss_sum']:.4f}",
        f"Test MAE: {test_metrics['mae_sum']:.4f}",
    ])
    plt.savefig(os.path.join(out_dir, "sum_prediction.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 5) One 2x2 figure like the sample image
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], marker="o")
    ax.plot(epochs, history["val_loss"], marker="s")
    ax.axhline(test_metrics["loss"], linestyle="--")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train Total Loss", "Val Total Loss", f"Test Loss: {test_metrics['loss']:.4f}"])

    ax = axes[0, 1]
    ax.plot(epochs, history["train_loss_missing"], marker="o")
    ax.plot(epochs, history["train_acc_missing"], marker="^")
    ax.plot(epochs, history["val_loss_missing"], marker="s")
    ax.plot(epochs, history["val_acc_missing"], marker="D")
    ax.axhline(test_metrics["loss_missing"], linestyle="--")
    ax.axhline(test_metrics["acc_missing"], linestyle="--")
    ax.set_title("Missing Digit Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.legend([
        "Train Loss", "Train Acc", "Val Loss", "Val Acc",
        f"Test Loss: {test_metrics['loss_missing']:.4f}",
        f"Test Acc: {test_metrics['acc_missing']:.4f}",
    ])

    ax = axes[1, 0]
    ax.plot(epochs, history["train_loss_sorted"], marker="o")
    ax.plot(epochs, history["train_acc_sorted"], marker="^")
    ax.plot(epochs, history["val_loss_sorted"], marker="s")
    ax.plot(epochs, history["val_acc_sorted"], marker="D")
    ax.axhline(test_metrics["loss_sorted"], linestyle="--")
    ax.axhline(test_metrics["acc_sorted"], linestyle="--")
    ax.set_title("Sort Order Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.legend([
        "Train Loss", "Train Acc", "Val Loss", "Val Acc",
        f"Test Loss: {test_metrics['loss_sorted']:.4f}",
        f"Test Acc: {test_metrics['acc_sorted']:.4f}",
    ])

    ax = axes[1, 1]
    ax.plot(epochs, history["train_loss_sum"], marker="o")
    ax.plot(epochs, history["train_mae_sum"], marker="^")
    ax.plot(epochs, history["val_loss_sum"], marker="s")
    ax.plot(epochs, history["val_mae_sum"], marker="D")
    ax.axhline(test_metrics["loss_sum"], linestyle="--")
    ax.axhline(test_metrics["mae_sum"], linestyle="--")
    ax.set_title("Sum Prediction Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.legend([
        "Train Loss", "Train MAE", "Val Loss", "Val MAE",
        f"Test Loss: {test_metrics['loss_sum']:.4f}",
        f"Test MAE: {test_metrics['mae_sum']:.4f}",
    ])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_history.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def train(
    root_dir="./data",
    epochs=15,
    batch_size=256,
    lr=3e-3,
    weight_decay=1e-4,
    out_dir="./outputs",
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    device = get_device()
    print("Device:", device)

    num_workers = 0 if device.type == "mps" else 2
    pin_memory = (device.type == "cuda")

    train_ds, val_ds, test_ds = get_datasets(root_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    model = SudokuNet().to(device)
    print("Trainable parameters:", count_params(model))

    # Loss functions
    ce_missing = nn.CrossEntropyLoss()

    # Handle class imbalance for sorted task (class 0 is ~4x more frequent)
    sorted_class_weights = torch.tensor([1.0, 4.0, 4.0], device=device)
    ce_sorted = nn.CrossEntropyLoss(weight=sorted_class_weights)

    reg_loss = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # We also track per-task losses (to make plots like the sample)
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc_missing": [], "val_acc_missing": [],
        "train_acc_sorted": [], "val_acc_sorted": [],
        "train_mae_sum": [], "val_mae_sum": [],

        "train_loss_missing": [], "val_loss_missing": [],
        "train_loss_sorted": [], "val_loss_sorted": [],
        "train_loss_sum": [], "val_loss_sum": [],
    }

    best_score = -1.0
    w_sorted = 2.0  # sorted loss weight

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        # Accumulate train losses and metrics
        total_loss = 0.0
        total_loss_missing = 0.0
        total_loss_sorted = 0.0
        total_loss_sum = 0.0

        correct_missing = 0
        total_missing = 0

        correct_sorted = 0
        total_sorted = 0

        mae_sum_total = 0.0
        sum_count = 0

        n_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y_missing = batch["missing_digit"].to(device)
            y_sorted = batch["sorted_labels"].to(device)
            y_sum = batch["sum_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)

            loss_missing = ce_missing(out["missing_logits"], y_missing)
            loss_sorted = ce_sorted(out["sorted_logits"].reshape(-1, 3), y_sorted.reshape(-1))
            loss_sum = reg_loss(out["sum_pred"], y_sum)

            loss = loss_missing + w_sorted * loss_sorted + loss_sum
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_missing += loss_missing.item()
            total_loss_sorted += loss_sorted.item()
            total_loss_sum += loss_sum.item()

            n_batches += 1

            # Train metrics
            correct_missing += (out["missing_logits"].argmax(1) == y_missing).sum().item()
            total_missing += y_missing.numel()

            correct_sorted += (out["sorted_logits"].argmax(2) == y_sorted).sum().item()
            total_sorted += y_sorted.numel()

            mae_sum_total += torch.abs(out["sum_pred"] - y_sum).sum().item()
            sum_count += y_sum.numel()

        train_metrics = {
            "loss": total_loss / max(1, n_batches),
            "loss_missing": total_loss_missing / max(1, n_batches),
            "loss_sorted": total_loss_sorted / max(1, n_batches),
            "loss_sum": total_loss_sum / max(1, n_batches),
            "acc_missing": correct_missing / max(1, total_missing),
            "acc_sorted": correct_sorted / max(1, total_sorted),
            "mae_sum": mae_sum_total / max(1, sum_count),
        }

        # Validate
        val_metrics = evaluate(model, val_loader, device, ce_missing, ce_sorted, reg_loss, w_sorted=w_sorted)

        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc_missing"].append(train_metrics["acc_missing"])
        history["train_acc_sorted"].append(train_metrics["acc_sorted"])
        history["train_mae_sum"].append(train_metrics["mae_sum"])

        history["train_loss_missing"].append(train_metrics["loss_missing"])
        history["train_loss_sorted"].append(train_metrics["loss_sorted"])
        history["train_loss_sum"].append(train_metrics["loss_sum"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_acc_missing"].append(val_metrics["acc_missing"])
        history["val_acc_sorted"].append(val_metrics["acc_sorted"])
        history["val_mae_sum"].append(val_metrics["mae_sum"])

        # We don't separately log per-task val losses inside evaluate right now,
        # so we compute them quickly here for plotting consistency.
        # (Cheap on val set, still fine.)
        val_loss_missing = 0.0
        val_loss_sorted = 0.0
        val_loss_sum = 0.0
        vb = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y_missing = batch["missing_digit"].to(device)
                y_sorted = batch["sorted_labels"].to(device)
                y_sum = batch["sum_labels"].to(device)

                out = model(x)
                val_loss_missing += ce_missing(out["missing_logits"], y_missing).item()
                val_loss_sorted += ce_sorted(out["sorted_logits"].reshape(-1, 3), y_sorted.reshape(-1)).item()
                val_loss_sum += reg_loss(out["sum_pred"], y_sum).item()
                vb += 1
        history["val_loss_missing"].append(val_loss_missing / max(1, vb))
        history["val_loss_sorted"].append(val_loss_sorted / max(1, vb))
        history["val_loss_sum"].append(val_loss_sum / max(1, vb))

        # Checkpoint best model
        score = 0.5 * val_metrics["acc_missing"] + 0.5 * val_metrics["acc_sorted"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | {elapsed:.1f}s | "
            f"val acc(miss) {val_metrics['acc_missing']:.3f} | "
            f"val acc(sort) {val_metrics['acc_sorted']:.3f} | "
            f"val mae(sum) {val_metrics['mae_sum']:.4f}"
        )

    # Final test (also compute per-task test losses for plots)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    model.to(device)
    model.eval()

    test_metrics_main = evaluate(model, test_loader, device, ce_missing, ce_sorted, reg_loss, w_sorted=w_sorted)

    # Per-task test losses
    test_loss_missing = 0.0
    test_loss_sorted = 0.0
    test_loss_sum = 0.0
    tb = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y_missing = batch["missing_digit"].to(device)
            y_sorted = batch["sorted_labels"].to(device)
            y_sum = batch["sum_labels"].to(device)

            out = model(x)
            test_loss_missing += ce_missing(out["missing_logits"], y_missing).item()
            test_loss_sorted += ce_sorted(out["sorted_logits"].reshape(-1, 3), y_sorted.reshape(-1)).item()
            test_loss_sum += reg_loss(out["sum_pred"], y_sum).item()
            tb += 1

    test_metrics = {
        **test_metrics_main,
        "loss_missing": test_loss_missing / max(1, tb),
        "loss_sorted": test_loss_sorted / max(1, tb),
        "loss_sum": test_loss_sum / max(1, tb),
    }

    print("\n[Test metrics]")
    print(test_metrics)

    # Save plots + a text file
    save_plots(history, test_metrics, out_dir)
    with open(os.path.join(out_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(str(test_metrics) + "\n")


if __name__ == "__main__":
    train()
