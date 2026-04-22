from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Compare deletion curves and emit report")
    p.add_argument("--resnet-csv", type=str, required=True)
    p.add_argument("--sd-csv", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/analysis_report.md")
    return p.parse_args()


def find_knee(df: pd.DataFrame) -> int:
    base_err = float(df.iloc[0]["val_error"])
    delta = df["val_error"] - base_err
    idx = delta.gt(0.5).idxmax()
    if bool(delta.gt(0.5).any()):
        return int(df.loc[idx, "deleted_blocks"])
    return int(df["deleted_blocks"].max())


def main() -> None:
    args = parse_args()
    a = pd.read_csv(args.resnet_csv)
    b = pd.read_csv(args.sd_csv)

    a0 = float(a.iloc[0]["val_error"])
    b0 = float(b.iloc[0]["val_error"])
    a_knee = find_knee(a)
    b_knee = find_knee(b)

    lines = []
    lines.append("# Deletion Curve Analysis")
    lines.append("")
    lines.append("## Core observations")
    lines.append(f"- Baseline validation error: resnet152={a0:.3f}, resnet152_sd={b0:.3f}.")
    lines.append(
        f"- Robust depth range under +0.5 error: resnet152 deletes ~{a_knee} blocks, "
        f"resnet152_sd deletes ~{b_knee} blocks."
    )
    lines.append(
        "- Typical pattern is flat early growth and sharp late growth; this means tail blocks can be partially redundant."
    )
    lines.append("")
    lines.append("## Why this happens")
    lines.append(
        "- Residual design preserves identity pathways, so early-to-mid features remain usable when some tail blocks are removed."
    )
    lines.append(
        "- Stochastic depth trains with random block dropping; at test time this yields stronger tolerance to explicit truncation."
    )
    lines.append(
        "- The final stage mostly refines semantic detail; removing too many blocks eventually removes discriminative capacity and error rises quickly."
    )
    lines.append("")
    lines.append("## Can this be used for acceleration?")
    lines.append(
        "- Yes. If a target error budget is defined, choose the largest tail truncation that stays below it."
    )
    lines.append(
        "- This gives a direct quality-speed knob and often better robustness for stochastic-depth models."
    )
    lines.append(
        "- Practical deployment should calibrate deletion count per hardware and batch size, then freeze active block count in serving."
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis report to {out}")


if __name__ == "__main__":
    main()
