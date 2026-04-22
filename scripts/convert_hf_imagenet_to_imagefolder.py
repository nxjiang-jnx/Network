from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Convert HF ILSVRC/imagenet-1k dataset to ImageFolder layout"
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="HF dataset repo id",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="./data/imagenet1k_imagefolder",
        help="Output ImageFolder root, will create train/ and val/",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="./data/hf_imagenet1k_raw/.cache",
        help="Datasets cache dir (optional)",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint (for mirror usage)",
    )
    p.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token (optional, fallback to HF_TOKEN env or ./.hf_token)",
    )
    p.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if image-format=jpg",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip file if already exists",
    )
    return p.parse_args()


def save_image(img: Image.Image, dst: Path, fmt: str, jpeg_quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpg":
        img.convert("RGB").save(dst, format="JPEG", quality=jpeg_quality)
    else:
        img.save(dst, format="PNG")


def convert_split(
    repo_id: str,
    split_name: str,
    split_out_dir: Path,
    cache_dir: str | None,
    token: str | None,
    image_format: str,
    jpeg_quality: int,
    skip_existing: bool,
) -> None:
    ds = load_dataset(
        repo_id,
        split=split_name,
        cache_dir=cache_dir or None,
        token=token,
    )

    total = len(ds)
    pbar = tqdm(range(total), desc=f"convert {split_name}")
    for i in pbar:
        ex = ds[i]
        label = int(ex["label"])
        img = ex["image"]
        cls_dir = split_out_dir / f"{label:04d}"
        suffix = "jpg" if image_format == "jpg" else "png"
        dst = cls_dir / f"{i:08d}.{suffix}"
        if skip_existing and dst.exists():
            continue
        save_image(img, dst, image_format, jpeg_quality)


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path("./.hf_token")
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
    output_root = Path(args.output_root).expanduser().resolve()
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    convert_split(
        repo_id=args.repo_id,
        split_name="train",
        split_out_dir=train_dir,
        cache_dir=args.cache_dir,
        token=token,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        skip_existing=args.skip_existing,
    )
    convert_split(
        repo_id=args.repo_id,
        split_name="validation",
        split_out_dir=val_dir,
        cache_dir=args.cache_dir,
        token=token,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        skip_existing=args.skip_existing,
    )
    print(f"Done. ImageFolder root: {output_root}")


if __name__ == "__main__":
    main()
