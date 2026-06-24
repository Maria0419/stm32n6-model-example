import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_utils import load_train_config, resolve_path
from dataset import SquareSegDataset
from model import DiceBCELoss, build_model


def compute_segmentation_metrics(logits, masks, threshold):
    predictions = (torch.sigmoid(logits) > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (predictions * masks).sum(dim=(1, 2, 3))
    prediction_sum = predictions.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = prediction_sum + mask_sum - intersection
    dice = ((2.0 * intersection + 1.0) / (prediction_sum + mask_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    return dice, iou


def run_epoch(model, loader, criterion, device, threshold, epoch, total_epochs, stage, optimizer=None, max_grad_norm=0.0):
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{stage}]", leave=False)

    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_training else torch.no_grad()
        with grad_context:
            logits = model(images)
            loss = criterion(logits, masks)
            if is_training:
                loss.backward()
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        logits = logits.detach().float()
        dice, iou = compute_segmentation_metrics(logits, masks, threshold)
        total_loss += float(loss.item())
        total_dice += dice
        total_iou += iou
        batch_count += 1
        progress.set_postfix(loss=f"{total_loss / batch_count:.4f}", dice=f"{total_dice / batch_count:.4f}", iou=f"{total_iou / batch_count:.4f}")

    progress.close()
    return {
        "loss": total_loss / max(1, batch_count),
        "dice": total_dice / max(1, batch_count),
        "iou": total_iou / max(1, batch_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the square_seg 32x13 model")
    parser.add_argument("--config", default="models/square_seg_stm32n6/configs/train.yaml")
    args = parser.parse_args()

    config_path, cfg = load_train_config(args.config)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    artifacts_dir = resolve_path(config_path, cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = {
        "image_height": cfg["image_height"],
        "image_width": cfg["image_width"],
    }
    train_dataset = SquareSegDataset(resolve_path(config_path, cfg["train_dir"]), augment=cfg.get("augment", False), **dataset_args)
    test_dataset = SquareSegDataset(resolve_path(config_path, cfg["test_dir"]), augment=False, **dataset_args)
    print(f"Training data: {len(train_dataset)}  Testing data: {len(test_dataset)}")

    loader_args = {
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, generator=generator, **loader_args)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]), shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    criterion = DiceBCELoss(
        bce_weight=cfg.get("bce_loss_weight", 1.0),
        pos_weight=cfg.get("pos_weight", 1.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    threshold = cfg.get("threshold", 0.5)
    max_epochs = cfg["epochs"]
    patience = max(1, cfg.get("early_stopping_patience", 5))
    max_grad_norm = cfg.get("max_grad_norm", 0.0)

    best_iou = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, threshold, epoch, max_epochs, "train", optimizer=optimizer, max_grad_norm=max_grad_norm)
        test_metrics = run_epoch(model, test_loader, criterion, device, threshold, epoch, max_epochs, "test")
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
            "threshold": threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        model_state = {
            "model_state": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "metrics": row,
            "history": history,
        }
        torch.save(model_state, artifacts_dir / "model.pt")

        if test_metrics["iou"] > best_iou:
            best_iou = float(test_metrics["iou"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch, "best_iou": best_iou, "patience": patience}, sort_keys=True), flush=True)
                break

    (artifacts_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
