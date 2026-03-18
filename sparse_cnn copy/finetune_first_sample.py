import argparse
import glob
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trail import (
    V2XRadarDataset,
    MultiModalDetectionNetwork,
    DetectionLoss,
    collate_fn,
    train_one_epoch,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = r"F:\Work\DeepLearning\Research\V2X-Radar-V"


def find_latest_checkpoint(checkpoint_dir: str):
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return None, None

    latest_epoch = -1
    latest_path = None
    for path in checkpoint_files:
        try:
            epoch = int(os.path.basename(path).split("_epoch_")[1].split(".pth")[0])
        except (IndexError, ValueError):
            continue
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path

    if latest_path is None:
        return None, None
    return latest_path, latest_epoch


def disable_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0


def freeze_batchnorm(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


def set_trainable_params(model: nn.Module, head_only: bool):
    if not head_only:
        for p in model.parameters():
            p.requires_grad_(True)
        return

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.detector.parameters():
        p.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune latest model on first LiDAR/radar/image sample."
    )
    parser.add_argument("--root-dir", default=DEFAULT_DATA_ROOT, help="Dataset root directory")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT),
        help="Directory containing checkpoint_epoch_*.pth",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "checkpoints"),
        help="Output directory for finetuned checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Finetuning epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also load optimizer state from latest checkpoint",
    )
    parser.add_argument(
        "--overfit-preset",
        action="store_true",
        help="Apply single-sample overfit settings: head-only + freeze-bn + disable-dropout",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Train only detection head (recommended for single-sample overfit)",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Keep BatchNorm layers in eval mode during finetuning",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="Set dropout probability to 0 during finetuning",
    )
    args = parser.parse_args()

    if args.overfit_preset:
        args.head_only = True
        args.freeze_bn = True
        args.disable_dropout = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    latest_ckpt, latest_epoch = find_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint_epoch_*.pth found in checkpoint directory: {args.checkpoint_dir}"
        )

    print(f"Using checkpoint: {latest_ckpt} (epoch {latest_epoch})")
    print(f"Device: {device}")

    dataset = V2XRadarDataset(args.root_dir, split="training", num_samples=1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = MultiModalDetectionNetwork(lidar_dim=128, radar_dim=128, image_dim=128).to(device)
    det_criterion = DetectionLoss()

    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.disable_dropout:
        disable_dropout(model)
    set_trainable_params(model, args.head_only)

    # Ensure model.train() (inside train_one_epoch) keeps BN frozen when requested.
    if args.freeze_bn:
        original_train = model.train

        def train_with_frozen_bn(mode: bool = True):
            result = original_train(mode)
            if mode:
                freeze_batchnorm(model)
            return result

        model.train = train_with_frozen_bn

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_optimizer and "optimizer_state_dict" in checkpoint and not args.head_only:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Starting single-sample finetuning on sample index 0...")
    print(
        f"Config: epochs={args.epochs}, lr={args.lr}, head_only={args.head_only}, "
        f"freeze_bn={args.freeze_bn}, disable_dropout={args.disable_dropout}"
    )
    losses = []
    start_epoch = (latest_epoch or 0) + 1
    for i in range(args.epochs):
        epoch_num = start_epoch + i
        loss, stats = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            det_criterion=det_criterion,
            device=device,
            epoch=epoch_num,
            root_dir=args.root_dir,
        )
        losses.append(loss)
        print(
            f"Epoch {epoch_num}: loss={loss:.4f}, "
            f"rpn_cls={stats['rpn_cls_loss']:.4f}, "
            f"rpn_reg={stats['rpn_reg_loss']:.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"finetuned_first_sample_from_epoch_{latest_epoch}_{timestamp}.pth"
    out_path = os.path.join(args.output_dir, out_name)

    torch.save(
        {
            "base_checkpoint": latest_ckpt,
            "base_epoch": latest_epoch,
            "finetune_epochs": args.epochs,
            "sample_index": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
        },
        out_path,
    )

    print(f"Saved finetuned checkpoint: {out_path}")


if __name__ == "__main__":
    main()
