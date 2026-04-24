from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

# Must run before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchvision import datasets

from models import build_resnet152, build_resnet152_stochastic_depth
from utils.data import build_imagenet_loaders
from utils.metrics import AverageMeter, accuracy

AMP_DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ImageNet training for ResNet-152 variants")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--model", type=str, choices=["resnet152", "resnet152_sd"], required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=112)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--lr-scheduler", type=str, choices=["multistep", "cosine"], default="multistep")
    p.add_argument("--lr-milestones", type=str, default="30,60,90")
    p.add_argument("--sd-p-last", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    return p.parse_args()


def parse_milestones(spec: str, epochs: int) -> list[int]:
    vals: list[int] = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if 0 < v < epochs:
            vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        vals = [30, 60, 90]
    return vals


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "resnet152":
        return build_resnet152(num_classes=1000)
    return build_resnet152_stochastic_depth(p_last=args.sd_p_last, num_classes=1000)


def unwrap_state_dict(model: nn.Module) -> dict:
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def load_state_dict(model: nn.Module, state: dict) -> None:
    if isinstance(model, DDP):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def eval_model_for_inference(model: nn.Module) -> nn.Module:
    """DDP requires all ranks to participate in forward; rank-0-only val must use .module."""
    return model.module if isinstance(model, DDP) else model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    accum = max(int(grad_accum_steps), 1)
    num_batches = len(loader)
    remainder = num_batches % accum

    pbar = tqdm(loader, desc=f"train {epoch + 1}/{epochs}", leave=False, disable=dist.is_initialized() and dist.get_rank() != 0)
    optimizer.zero_grad(set_to_none=True)
    for step_idx, (images, target) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)

        is_update_step = (step_idx % accum == 0) or (step_idx == num_batches)
        current_accum = remainder if (step_idx == num_batches and remainder != 0) else accum

        sync_ctx = nullcontext()
        if isinstance(model, DDP) and not is_update_step:
            sync_ctx = model.no_sync()

        with sync_ctx:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                output = model(images)
                loss = criterion(output, target)
            (loss / current_accum).backward()

        if is_update_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(float(loss.item()), bs)
        top1_meter.update(float(acc1.item()), bs)
        top5_meter.update(float(acc5.item()), bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}")

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
        "_loss_sum": loss_meter.sum,
        "_loss_cnt": loss_meter.count,
        "_top1_sum": top1_meter.sum,
        "_top1_cnt": top1_meter.count,
        "_top5_sum": top5_meter.sum,
        "_top5_cnt": top5_meter.count,
    }


def sync_train_stats(stats: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized():
        return {k: v for k, v in stats.items() if not k.startswith("_")}
    t = torch.tensor(
        [
            stats["_loss_sum"],
            stats["_loss_cnt"],
            stats["_top1_sum"],
            stats["_top1_cnt"],
            stats["_top5_sum"],
            stats["_top5_cnt"],
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ls, lc, u1, c1, u5, c5 = t.tolist()
    return {
        "loss": float(ls / max(lc, 1)),
        "top1": float(u1 / max(c1, 1)),
        "top5": float(u5 / max(c5, 1)),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    for images, target in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(float(loss.item()), bs)
        top1_meter.update(float(acc1.item()), bs)
        top5_meter.update(float(acc5.item()), bs)
    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
        "_loss_sum": loss_meter.sum,
        "_loss_cnt": loss_meter.count,
        "_top1_sum": top1_meter.sum,
        "_top1_cnt": top1_meter.count,
        "_top5_sum": top5_meter.sum,
        "_top5_cnt": top5_meter.count,
    }


def sync_eval_stats(stats: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized():
        return {k: v for k, v in stats.items() if not k.startswith("_")}
    t = torch.tensor(
        [
            stats["_loss_sum"],
            stats["_loss_cnt"],
            stats["_top1_sum"],
            stats["_top1_cnt"],
            stats["_top5_sum"],
            stats["_top5_cnt"],
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ls, lc, u1, c1, u5, c5 = t.tolist()
    return {
        "loss": float(ls / max(lc, 1)),
        "top1": float(u1 / max(c1, 1)),
        "top5": float(u5 / max(c5, 1)),
    }


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()

    if distributed:
        device = torch.device("cuda", local_rank)
        lr = args.lr * world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = args.lr

    # Keep model/data shuffling deterministic per-rank while SD gate sampling is synchronized by broadcast.
    set_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir) / args.model
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    ckpt_last = output_dir / "last.pt"
    ckpt_best = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"

    model = create_model(args).to(device).to(memory_format=torch.channels_last)
    if distributed:
        # Stochastic depth can skip whole blocks, so parameter usage is dynamic per step.
        use_sd = args.model == "resnet152_sd"
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=not use_sd,
            find_unused_parameters=use_sd,
            gradient_as_bucket_view=True,
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.lr_scheduler == "multistep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=parse_milestones(args.lr_milestones, args.epochs),
            gamma=0.1,
        )
    else:
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
        load_state_dict(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_top1 = float(state.get("best_top1", 0.0))
        if rank == 0:
            print(f"Resumed from {resume_path}, start_epoch={start_epoch}")

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if distributed:
        train_ds = datasets.ImageFolder(str(Path(args.data_root) / "train"))
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_ds = datasets.ImageFolder(str(Path(args.data_root) / "val"))
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        build_val=True,
        prefetch_factor=args.prefetch_factor,
    )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_raw = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
        )
        train_stats = sync_train_stats(train_raw, device)

        val_raw = evaluate(
            model=eval_model_for_inference(model),
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_stats = sync_eval_stats(val_raw, device)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train": train_stats,
            "val": val_stats,
            "val_error": 100.0 - val_stats["top1"],
        }
        if rank == 0:
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            is_best = val_stats["top1"] > best_top1
            best_top1 = max(best_top1, val_stats["top1"])
            save_state = {
                "epoch": epoch,
                "model": unwrap_state_dict(model),
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

    if rank == 0:
        print(f"Training finished. Best val top1={best_top1:.2f}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
