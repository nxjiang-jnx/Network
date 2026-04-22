from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
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

    def _run_block_train(self, x: torch.Tensor, block: nn.Module, p: float) -> torch.Tensor:
        if p >= 1.0:
            return block(x)
        if torch.rand(1, device=x.device).item() < p:
            return block(x)
        return x

    def _run_block_eval(self, x: torch.Tensor, block: nn.Module, p: float) -> torch.Tensor:
        if p >= 1.0:
            return block(x)
        out = block(x)
        identity = x
        residual = out - identity
        return identity + p * residual

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
