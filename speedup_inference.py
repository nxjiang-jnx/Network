from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch

from models import build_resnet152, build_resnet152_stochastic_depth
from utils.data import build_imagenet_loaders, device_from_flag
from utils.metrics import accuracy

AMP_DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inference speedup by tail block truncation")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["resnet152", "resnet152_sd"], required=True)
    p.add_argument("--sd-p-last", type=float, default=0.5)
    p.add_argument("--error-budget", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--prefetch-factor", type=int, default=6)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--bench-iters", type=int, default=80)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="outputs/accel")
    return p.parse_args()


def create_model(args: argparse.Namespace):
    if args.model == "resnet152":
        return build_resnet152(num_classes=1000)
    return build_resnet152_stochastic_depth(p_last=args.sd_p_last, num_classes=1000)


@torch.no_grad()
def eval_top1(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0.0
    total = 0
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
        if device.type == "cuda"
        else nullcontext()
    )
    for images, target in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        with amp_ctx:
            out = model(images)
        acc1 = accuracy(out, target, topk=(1,))[0].item()
        bs = images.shape[0]
        correct += (acc1 / 100.0) * bs
        total += bs
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def benchmark_throughput(model, device: torch.device, batch_size: int, warmup: int, iters: int) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
        if device.type == "cuda"
        else nullcontext()
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        with amp_ctx:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with amp_ctx:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (batch_size * iters) / max(elapsed, 1e-12)


def main() -> None:
    args = parse_args()
    device = device_from_flag(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = build_imagenet_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = create_model(args).to(device).to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    total_blocks = model.total_blocks

    base_top1 = None
    base_thr = None
    rows: List[dict] = []
    chosen = None

    for deleted in range(0, total_blocks):
        active = total_blocks - deleted
        model.set_active_block_count(active)
        top1 = eval_top1(model, val_loader, device)
        err = 100.0 - top1
        thr = benchmark_throughput(
            model,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup_iters,
            iters=args.bench_iters,
        )
        if base_top1 is None:
            base_top1 = top1
            base_thr = thr
        delta_err = (100.0 - top1) - (100.0 - base_top1)
        speedup = thr / max(base_thr, 1e-12)
        row = {
            "deleted_blocks": deleted,
            "active_blocks": active,
            "top1": top1,
            "val_error": err,
            "delta_error_vs_full": delta_err,
            "throughput_img_s": thr,
            "speedup_vs_full": speedup,
        }
        rows.append(row)
        print(
            f"[{args.model}] deleted={deleted:02d} top1={top1:.3f} delta_err={delta_err:.3f} "
            f"thr={thr:.1f} img/s speedup={speedup:.3f}x"
        )
        if delta_err <= args.error_budget:
            chosen = row
        else:
            break

    csv_path = out_dir / "accel_tradeoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_path = out_dir / "recommended.txt"
    if chosen is None:
        msg = "No truncation satisfies the error budget."
    else:
        msg = (
            f"Recommended deleted_blocks={int(chosen['deleted_blocks'])}, "
            f"active_blocks={int(chosen['active_blocks'])}, "
            f"delta_error={chosen['delta_error_vs_full']:.4f}, "
            f"speedup={chosen['speedup_vs_full']:.4f}x"
        )
    best_path.write_text(msg + "\n", encoding="utf-8")
    print(f"[{args.model}] {msg}")
    print(f"[{args.model}] Saved {csv_path}")
    print(f"[{args.model}] Saved {best_path}")


if __name__ == "__main__":
    main()
