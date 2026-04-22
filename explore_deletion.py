from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from models import build_resnet152, build_resnet152_stochastic_depth
from utils.data import build_imagenet_loaders, device_from_flag
from utils.metrics import accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Back-to-front layer deletion study")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["resnet152", "resnet152_sd"], required=True)
    p.add_argument("--sd-p-last", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-delete", type=int, default=-1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="outputs/explore")
    return p.parse_args()


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "resnet152":
        return build_resnet152(num_classes=1000)
    return build_resnet152_stochastic_depth(p_last=args.sd_p_last, num_classes=1000)


@torch.no_grad()
def evaluate_top1(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0.0
    total = 0
    for images, target in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(images)
        acc1 = accuracy(out, target, topk=(1,))[0].item()
        bs = images.size(0)
        correct += (acc1 / 100.0) * bs
        total += bs
    return (correct / max(total, 1)) * 100.0


def main() -> None:
    args = parse_args()
    device = device_from_flag(args.device)
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        workers=args.workers,
    )

    model = build_model(args).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    total_blocks = model.total_blocks
    max_delete = total_blocks if args.max_delete < 0 else min(args.max_delete, total_blocks - 1)

    rows: List[Dict[str, float]] = []
    for deleted in range(0, max_delete + 1, args.step):
        active = total_blocks - deleted
        model.set_active_block_count(active)
        top1 = evaluate_top1(model, val_loader, device)
        error = 100.0 - top1
        rows.append(
            {
                "deleted_blocks": deleted,
                "active_blocks": active,
                "top1": top1,
                "val_error": error,
            }
        )
        print(
            f"deleted={deleted:02d} active={active:02d}/{total_blocks} "
            f"top1={top1:.3f} val_error={error:.3f}"
        )

    csv_path = out_dir / "deletion_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [r["deleted_blocks"] for r in rows]
    y = [r["val_error"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1.2)
    plt.xlabel("Deleted residual blocks (from tail)")
    plt.ylabel("Validation error (%)")
    plt.title(f"Deletion curve: {args.model}")
    plt.grid(True, linestyle="--", alpha=0.4)
    fig_path = out_dir / "deletion_curve.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
