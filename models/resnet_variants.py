from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as tv_models


@dataclass
class BlockMeta:
    stage_name: str
    block_idx_in_stage: int
    global_idx: int
    survival_prob: float


class StochasticDepthResNet(nn.Module):
    def __init__(self, p_last: float = 0.5, num_classes: int = 1000) -> None:
        super().__init__()
        if not 0.0 < p_last <= 1.0:
            raise ValueError("p_last must be in (0, 1].")

        base = tv_models.resnet152(weights=None, num_classes=num_classes)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.avgpool = base.avgpool
        self.fc = base.fc

        self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        self.stage_names = ["layer1", "layer2", "layer3", "layer4"]
        self.blocks = self._flatten_blocks(self.stages)
        self.total_blocks = len(self.blocks)
        self.p_last = p_last
        self.block_meta = self._make_block_meta()
        self._is_bottleneck_block = [self._is_torchvision_bottleneck(b) for b in self.blocks]
        self.register_buffer(
            "_survival_probs",
            torch.tensor([m.survival_prob for m in self.block_meta], dtype=torch.float32),
            persistent=False,
        )
        self.active_block_count = self.total_blocks

    @staticmethod
    def _flatten_blocks(stages: Iterable[nn.Sequential]) -> nn.ModuleList:
        flat: List[nn.Module] = []
        for stage in stages:
            flat.extend(list(stage.children()))
        return nn.ModuleList(flat)

    def _make_block_meta(self) -> List[BlockMeta]:
        metas: List[BlockMeta] = []
        global_idx = 0
        for stage_name, stage in zip(self.stage_names, self.stages):
            for block_idx, _ in enumerate(stage):
                l = global_idx + 1
                L = self.total_blocks
                p = 1.0 - (l / L) * (1.0 - self.p_last)
                metas.append(
                    BlockMeta(
                        stage_name=stage_name,
                        block_idx_in_stage=block_idx,
                        global_idx=global_idx,
                        survival_prob=float(p),
                    )
                )
                global_idx += 1
        return metas

    def set_active_block_count(self, count: int) -> None:
        if count < 1 or count > self.total_blocks:
            raise ValueError(f"active block count must be in [1, {self.total_blocks}]")
        self.active_block_count = count

    def get_all_survival_probs(self) -> List[float]:
        return [x.survival_prob for x in self.block_meta]

    @staticmethod
    def _is_torchvision_bottleneck(block: nn.Module) -> bool:
        return all(
            hasattr(block, name)
            for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu")
        )

    @staticmethod
    def _bottleneck_residual_and_identity(
        block: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = x if getattr(block, "downsample", None) is None else block.downsample(x)
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)
        out = block.conv3(out)
        out = block.bn3(out)
        return out, identity

    def _run_block_train(
        self, x: torch.Tensor, block: nn.Module, keep: bool, is_bottleneck: bool
    ) -> torch.Tensor:
        """Huang et al. CVPR'16: keep residual branch or bypass to shortcut."""
        if not is_bottleneck:
            return block(x) if keep else x
        if not keep:
            identity = x if getattr(block, "downsample", None) is None else block.downsample(x)
            return torch.relu(identity)
        residual, identity = self._bottleneck_residual_and_identity(block, x)
        return torch.relu(residual + identity)

    def _run_block_eval(
        self, x: torch.Tensor, block: nn.Module, p: float, is_bottleneck: bool
    ) -> torch.Tensor:
        # Paper-consistent test-time expectation: H_l = ReLU(f_l * p_l + h_l).
        if not is_bottleneck:
            return block(x)
        residual, identity = self._bottleneck_residual_and_identity(block, x)
        return torch.relu(residual * p + identity)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        keep_mask: Optional[list[bool]] = None
        active = self.active_block_count
        if self.training:
            probs = self._survival_probs[:active]
            if dist.is_initialized() and dist.get_world_size() > 1:
                mask = torch.empty(active, device=x.device, dtype=torch.bool)
                if dist.get_rank() == 0:
                    mask.copy_(torch.rand(active, device=x.device) < probs)
                dist.broadcast(mask, src=0)
            else:
                mask = torch.rand(active, device=x.device) < probs
            # Avoid per-block .item() host sync; sync once per step.
            keep_mask = mask.cpu().tolist()

        for i in range(active):
            block = self.blocks[i]
            is_bottleneck = self._is_bottleneck_block[i]
            if self.training:
                assert keep_mask is not None
                x = self._run_block_train(x, block, keep=keep_mask[i], is_bottleneck=is_bottleneck)
            else:
                p = self.block_meta[i].survival_prob
                x = self._run_block_eval(x, block, p=p, is_bottleneck=is_bottleneck)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TruncatableResNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        base = tv_models.resnet152(weights=None, num_classes=num_classes)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.avgpool = base.avgpool
        self.fc = base.fc
        self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        self.blocks = nn.ModuleList(
            [block for stage in self.stages for block in stage]
        )
        self.total_blocks = len(self.blocks)
        self.active_block_count = self.total_blocks

    def set_active_block_count(self, count: int) -> None:
        if count < 1 or count > self.total_blocks:
            raise ValueError(f"active block count must be in [1, {self.total_blocks}]")
        self.active_block_count = count

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks[: self.active_block_count]:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet152(num_classes: int = 1000) -> TruncatableResNet:
    return TruncatableResNet(num_classes=num_classes)


def build_resnet152_stochastic_depth(
    p_last: float = 0.5, num_classes: int = 1000
) -> StochasticDepthResNet:
    return StochasticDepthResNet(p_last=p_last, num_classes=num_classes)
