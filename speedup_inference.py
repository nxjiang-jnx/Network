from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from models import build_resnet152_stochastic_depth
from utils.data import device_from_flag

AMP_DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Minimal SD inference accelerator selector")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--deletion-csv", type=str, required=True)
    p.add_argument("--target-top1", type=float, default=75.0)
    p.add_argument("--sd-p-last", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-iters", type=int, default=30)
    p.add_argument("--bench-iters", type=int, default=120)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="outputs/accel/resnet152_sd")
    return p.parse_args()


def load_curve(path: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "deleted_blocks": int(r["deleted_blocks"]),
                    "active_blocks": int(r["active_blocks"]),
                    "top1": float(r["top1"]),
                    "val_error": float(r["val_error"]),
                }
            )
    if not rows:
        raise ValueError(f"deletion curve is empty: {path}")
    rows.sort(key=lambda x: x["deleted_blocks"])
    return rows


def choose_by_target_top1(rows: list[dict[str, float]], target_top1: float) -> dict[str, float]:
    feasible = [r for r in rows if r["top1"] >= target_top1]
    if not feasible:
        return rows[0]
    return feasible[-1]


@torch.no_grad()
def benchmark_throughput(
    model: torch.nn.Module,
    active_blocks: int,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iters: int,
) -> float:
    model.set_active_block_count(active_blocks)
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_curve(args.deletion_csv)
    base = rows[0]
    chosen = choose_by_target_top1(rows, args.target_top1)

    device = device_from_flag(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_resnet152_stochastic_depth(p_last=args.sd_p_last, num_classes=1000)
    model = model.to(device).to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)

    base_thr = benchmark_throughput(
        model=model,
        active_blocks=int(base["active_blocks"]),
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup_iters,
        iters=args.bench_iters,
    )
    chosen_thr = benchmark_throughput(
        model=model,
        active_blocks=int(chosen["active_blocks"]),
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup_iters,
        iters=args.bench_iters,
    )
    speedup = chosen_thr / max(base_thr, 1e-12)

    summary = {
        "target_top1": args.target_top1,
        "baseline": {
            "deleted_blocks": int(base["deleted_blocks"]),
            "active_blocks": int(base["active_blocks"]),
            "top1": float(base["top1"]),
            "val_error": float(base["val_error"]),
            "throughput_img_s": float(base_thr),
        },
        "chosen": {
            "deleted_blocks": int(chosen["deleted_blocks"]),
            "active_blocks": int(chosen["active_blocks"]),
            "top1": float(chosen["top1"]),
            "val_error": float(chosen["val_error"]),
            "throughput_img_s": float(chosen_thr),
            "delta_top1_vs_full": float(chosen["top1"] - base["top1"]),
            "delta_error_vs_full": float(chosen["val_error"] - base["val_error"]),
            "speedup_vs_full": float(speedup),
        },
    }

    throughput_csv = out_dir / "throughput_compare.csv"
    with throughput_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "deleted_blocks",
                "active_blocks",
                "top1",
                "val_error",
                "throughput_img_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "baseline",
                "deleted_blocks": int(base["deleted_blocks"]),
                "active_blocks": int(base["active_blocks"]),
                "top1": float(base["top1"]),
                "val_error": float(base["val_error"]),
                "throughput_img_s": float(base_thr),
            }
        )
        writer.writerow(
            {
                "name": "chosen",
                "deleted_blocks": int(chosen["deleted_blocks"]),
                "active_blocks": int(chosen["active_blocks"]),
                "top1": float(chosen["top1"]),
                "val_error": float(chosen["val_error"]),
                "throughput_img_s": float(chosen_thr),
            }
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "deploy_config.json").write_text(
        json.dumps(
            {
                "model": "resnet152_sd",
                "active_blocks": int(chosen["active_blocks"]),
                "deleted_blocks": int(chosen["deleted_blocks"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    msg = (
        f"Target Top1>={args.target_top1:.2f}, recommend deleting {int(chosen['deleted_blocks'])} layers "
        f"(active={int(chosen['active_blocks'])}/50), "
        f"Top1={float(chosen['top1']):.3f}, "
        f"Speedup={speedup:.4f}x."
    )
    (out_dir / "recommended.txt").write_text(msg + "\n", encoding="utf-8")
    print(msg)
    print(f"Saved {throughput_csv}")
    print(f"Saved {out_dir / 'summary.json'}")
    print(f"Saved {out_dir / 'deploy_config.json'}")
    print(f"Saved {out_dir / 'recommended.txt'}")


if __name__ == "__main__":
    main()
