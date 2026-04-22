# ResNet-152 and Stochastic Depth on ImageNet

This repository reproduces:

1. ResNet-152 and Stochastic-Depth ResNet-152 training on ImageNet for 120 epochs.
2. Back-to-front residual block deletion experiments on both models.
3. Validation-error vs deleted-blocks curves.
4. Inference acceleration by tail truncation under an error budget.

## Project layout

- `models/resnet_variants.py`: ResNet-152 and stochastic-depth variants with truncation support.
- `train_imagenet.py`: 120-epoch training script.
- `explore_deletion.py`: deletion curve generation.
- `speedup_inference.py`: accuracy-speed tradeoff calibration.
- `analyze_results.py`: automatic markdown report from experiment CSVs.
- `scripts/run_train_120e.sh`: one-shot training launcher.
- `scripts/run_explore_and_accel.sh`: one-shot exploration + acceleration launcher.

## Quick start

Install:

```bash
pip install -r requirements.txt
```

Expected ImageNet folder:

```text
<imagenet_root>/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

Train both models:

```bash
bash scripts/run_train_120e.sh /path/to/imagenet outputs cuda
```

Run deletion study + acceleration:

```bash
bash scripts/run_explore_and_accel.sh /path/to/imagenet outputs cuda
```

Main outputs:

- `outputs/resnet152/best.pt`
- `outputs/resnet152_sd/best.pt`
- `outputs/explore/resnet152/deletion_curve.csv`
- `outputs/explore/resnet152_sd/deletion_curve.csv`
- `outputs/analysis_report.md`
- `outputs/accel/resnet152/accel_tradeoff.csv`
- `outputs/accel/resnet152_sd/accel_tradeoff.csv`

## Notes on implementation

- ResNet-152 uses canonical stage depths `[3, 8, 36, 3]`.
- Stochastic depth uses linear survival schedule:
  `p_l = 1 - (l / L) * (1 - p_last)`, with default `p_last=0.5`.
- During training, each block is dropped with Bernoulli sampling.
- During eval, residual output is expectation-scaled by `p_l`.
- Tail truncation is implemented by `set_active_block_count`.

## Citation basis

- ResNet: He et al., "Deep Residual Learning for Image Recognition", 2015.
- Stochastic Depth: Huang et al., "Deep Networks with Stochastic Depth", 2016.
