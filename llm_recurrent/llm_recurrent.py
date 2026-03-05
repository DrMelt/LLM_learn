import sys
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F
import pickle


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from shared.units import *
from shared.model_env import TrainEnv
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


def train():
    current_dir = Path(__file__).resolve().parent
    torch.set_float32_matmul_precision("high")
    # hyperparameters
    load_model = False
    model_path = current_dir / Path("model/model.pth")
    # ------------
    train_env = TrainEnv()
    train_env.setup_tensorboard(Path("/root/tf-logs/llm_fixed"))
    train_env.setup_optimizers(
        learning_rate=2e-4,
        warmup_iters=5000,
        cos_T_0=4096,
        cos_T_mult=2,
        cos_eta_min=1e-6,
    )

    if load_model == False:
        with (current_dir / Path("../data_buffer/character_mapper.pkl")).open(
            "rb"
        ) as f:
            character_mapper = pickle.load(f)
        model = LLM_Recurrent(
            character_mapper,
            out_nums=4,
        )
        train_env.set_model(model)
    else:
        train_env.load_model(model_path=model_path)

    train_env.model_summary(input_size=(1, 512))

    train_env.load_data(
        current_dir / Path("../data_buffer/train_data.pt"),
        current_dir / Path("../data_buffer/val_data.pt"),
    )

    train_env.train_loop(
        save_iters=10000,
        eval_interval=1000,
        batch_size=256,
        eval_iters=10,
        max_data_len=512,
        save_dir=current_dir / Path("model"),
    )


if __name__ == "__main__":
    train()
