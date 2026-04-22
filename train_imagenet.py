from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from models import build_resnet152, build_resnet152_stochastic_depth
from utils.data import build_imagenet_loaders, device_from_flag
from utils.metrics import AverageMeter, accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ImageNet training for ResNet-152 variants")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--model", type=str, choices=["resnet152", "resnet152_sd"], required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--sd-p-last", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "resnet152":
        return build_resnet152(num_classes=1000)
    return build_resnet152_stochastic_depth(p_last=args.sd_p_last, num_classes=1000)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"train {epoch + 1}/{epochs}", leave=False)
    for images, target in pbar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(float(loss.item()), bs)
        top1_meter.update(float(acc1.item()), bs)
        top5_meter.update(float(acc5.item()), bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}")

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    for images, target in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(float(loss.item()), bs)
        top1_meter.update(float(acc1.item()), bs)
        top5_meter.update(float(acc5.item()), bs)
    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_flag(args.device)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = output_dir / "last.pt"
    ckpt_best = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"

    model = create_model(args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(args.warmup_epochs, 1))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )

    start_epoch = 0
    best_top1 = 0.0
    resume_path = Path(args.resume) if args.resume else ckpt_last
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_top1 = float(state.get("best_top1", 0.0))
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}")

    train_loader, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
    )

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )
        val_stats = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train": train_stats,
            "val": val_stats,
            "val_error": 100.0 - val_stats["top1"],
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        is_best = val_stats["top1"] > best_top1
        best_top1 = max(best_top1, val_stats["top1"])
        save_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_top1": best_top1,
            "args": vars(args),
        }
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(save_state, ckpt_last)
        if is_best:
            torch.save(save_state, ckpt_best)

        print(
            f"epoch={epoch + 1} lr={cur_lr:.6f} "
            f"train_top1={train_stats['top1']:.2f} val_top1={val_stats['top1']:.2f} "
            f"best_top1={best_top1:.2f}"
        )

    print(f"Training finished. Best val top1={best_top1:.2f}")


if __name__ == "__main__":
    main()
