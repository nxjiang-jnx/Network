from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Convert HF ILSVRC/imagenet-1k dataset to ImageFolder layout"
    )
    p.add_argument(
        "--parquet-dir",
        type=str,
        default="./data/hf_imagenet1k_raw/data",
        help="Directory containing train-*.parquet and validation-*.parquet",
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
        help="Datasets cache dir for parquet reader",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="Unused in offline parquet mode, kept for compatibility",
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
    parquet_files: list[str],
    split_name: str,
    split_out_dir: Path,
    cache_dir: str | None,
    image_format: str,
    jpeg_quality: int,
    skip_existing: bool,
) -> None:
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split={split_name}.")

    ds = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=cache_dir or None,
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

    output_root = Path(args.output_root).expanduser().resolve()
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    train_files = sorted(glob(str(parquet_dir / "train-*.parquet")))
    val_files = sorted(glob(str(parquet_dir / "validation-*.parquet")))

    if not train_files:
        raise FileNotFoundError(f"Missing train parquet files under {parquet_dir}")
    if not val_files:
        raise FileNotFoundError(f"Missing validation parquet files under {parquet_dir}")

    convert_split(
        parquet_files=train_files,
        split_name="train",
        split_out_dir=train_dir,
        cache_dir=args.cache_dir,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        skip_existing=args.skip_existing,
    )
    convert_split(
        parquet_files=val_files,
        split_name="validation",
        split_out_dir=val_dir,
        cache_dir=args.cache_dir,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        skip_existing=args.skip_existing,
    )
    print(f"Done. ImageFolder root: {output_root}")


if __name__ == "__main__":
    main()
