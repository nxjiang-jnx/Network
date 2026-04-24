from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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

    def _run_block_train(self, x: torch.Tensor, block: nn.Module, p: float) -> torch.Tensor:
        """Huang et al. CVPR'16: with prob p keep residual branch, else bypass to shortcut."""
        if p >= 1.0:
            return block(x)
        # DDP: one Bernoulli per block per forward, broadcast so all ranks share the same graph.
        if dist.is_initialized() and dist.get_world_size() > 1:
            u = torch.empty(1, device=x.device, dtype=torch.float32)
            if dist.get_rank() == 0:
                u.uniform_(0.0, 1.0)
            dist.broadcast(u, src=0)
            keep = u.item() < p
        else:
            keep = torch.rand(1, device=x.device).item() < p
        if not self._is_torchvision_bottleneck(block):
            return block(x) if keep else x
        residual, identity = self._bottleneck_residual_and_identity(block, x)
        if keep:
            return F.relu(residual + identity, inplace=False)
        return F.relu(identity, inplace=False)

    def _run_block_eval(self, x: torch.Tensor, block: nn.Module, p: float) -> torch.Tensor:
        # Paper-consistent test-time expectation: H_l = ReLU(f_l * p_l + h_l).
        if not self._is_torchvision_bottleneck(block):
            return block(x)
        residual, identity = self._bottleneck_residual_and_identity(block, x)
        return F.relu(residual * p + identity, inplace=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.blocks[: self.active_block_count]):
            p = self.block_meta[i].survival_prob
            if self.training:
                x = self._run_block_train(x, block, p)
            else:
                x = self._run_block_eval(x, block, p)
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
