import os
from pathlib import Path
import random
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import pickle

from shared.units import *
from shared.module import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class LLM_Recurrent(LLM_ModelBase):
    def __init__(
        self,
        vocab_map: CharacterMapper,
        out_nums: int,
    ):
        super().__init__(
            vocab_map=vocab_map,
            out_nums=out_nums,
        )
        # TODO: 添加你的模型初始化代码
        # 例如：定义嵌入层、RNN 层、输出层等

    def _target_offset_func(self, block_size: int) -> int:
        return super()._target_offset_func(block_size)

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1
    ) -> torch.Tensor:
        return super().generate(idx, max_new_tokens, temperature)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        return x

    def training_step(self, batch: torch.Tensor, target: torch.Tensor):
        pass


if __name__ == "__main__":
    pass
