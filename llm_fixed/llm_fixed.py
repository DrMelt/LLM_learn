import sys
from pathlib import Path
import random
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from shared import module, units, model_env
from shared.units import *
from shared.module import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class LLMFixedBlock(nn.Module):
    def __init__(self, a_embd: int, b_embd: int, head_nums: int, head_size: int):
        super().__init__()
        self.adaptive_norm = ForgetModule(vec_dim=b_embd)
        self.block_a2b = module.Block_A2B(
            a_embd=a_embd,
            b_embd=b_embd,
            n_head=head_nums,
            head_size=head_size,
        )
        self.block_self = module.Block_Self(
            n_embd=b_embd, n_head=head_nums, head_size=head_size
        )

    def forward(self, a, b):
        b = self.adaptive_norm(b)
        b = self.block_a2b(a, b)
        b = self.block_self(b)
        return b


class LLMFixedBlocks(nn.Module):
    def __init__(
        self, n_layer: int, head_nums: int, head_size: int, a_embd: int, b_embd: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LLMFixedBlock(
                    a_embd=a_embd,
                    b_embd=b_embd,
                    head_nums=head_nums,
                    head_size=head_size,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, a, b):
        for layer in self.layers:
            b = layer(a, b)
        return b


class LLMFixedModel(module.LLM_ModelBase):
    def __init__(
        self,
        vocab_map: units.CharacterMapper,
        token_embd: int,
        head_nums: int,
        head_size: int,
        n_layer: int,
        infer_vec_nums: int,
        infer_dim: int,
        forecast_steps: int,
    ):
        super().__init__(
            vocab_map=vocab_map,
            out_nums=forecast_steps,
        )
        self.out_nums = forecast_steps
        self.token_embd = token_embd
        self.info_n_embd = infer_dim
        self.infer_vec_nums = infer_vec_nums
        self.register_buffer("infer_arange", torch.arange(infer_vec_nums))
        self.infer_vec_embedder = nn.Embedding(infer_vec_nums, infer_dim)
        # each token directly reads off the logits for the next token from a lookup table
        self.embedder = module.CharacterEmbedder(
            vocab_size=vocab_map.max_vocab_size, n_embd=token_embd
        )

        self.blocks = LLMFixedBlocks(
            n_layer=n_layer,
            head_nums=head_nums,
            head_size=head_size,
            a_embd=token_embd,
            b_embd=infer_dim,
        )

        self.vec_to_word = module.Vec2Word(
            infer_embd_dim=infer_dim,
            token_dim=token_embd,
            vocab_size=vocab_map.max_vocab_size,
            forecast_steps=forecast_steps,
        )

        self.register_buffer(
            "infer_vec_weight",
            torch.arange(1, self.out_nums + 1, device=device).flip(0).view(1, -1)
            / self.out_nums,
        )  # (1, out_nums)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _target_offset_func(self, block_size: int) -> int:
        return block_size

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, _ = idx.shape

        inference_vec = self.infer_vec_embedder(
            self.get_buffer("infer_arange")
        )  # (info_vec_size, info_n_embd)
        inference_vec = inference_vec.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, info_vec_size, info_n_embd)
        token_vec = self.embedder.embed(idx)  # (B,T,C)

        inference_vec = self.blocks(
            token_vec, inference_vec
        )  # (B, info_vec_size, info_n_embd)
        logits = self.vec_to_word(inference_vec)  # (B, vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * self.out_nums, -1)  #  (B * out_nums, vocab_size)
            targets = targets.view(-1)  # (B * out_nums,)
            loss = F.cross_entropy(logits, targets, reduction="none").view(
                B, -1
            )  # (B, out_nums)
            loss_weight = self.get_buffer("infer_vec_weight")  # (1, out_nums)
            loss = (loss * loss_weight).mean()

        return logits, loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # # crop idx to the last block_size tokens
            # idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx)
            # apply softmax to get probabilities
            logits = logits[:, 0, :]  # (B, vocab_size)
            probs = F.softmax(logits / temperature, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def train_step(
        self,
        data: list[torch.Tensor],
        max_data_len: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        data_sample_len = random.randint(1, max_data_len)
        xb, yb = units.get_batch(
            data=data,
            block_size=data_sample_len,
            batch_size=batch_size,
            target_len=self.out_nums,
            target_offset=data_sample_len,
        )
        _, loss = self(xb, yb)
        loss = loss * (data_sample_len / (data_sample_len + 8.0))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()  # 更新学习率
        self.iter_n += 1


def train():
    # torch.set_float32_matmul_precision("high")
    current_dir = Path(__file__).resolve().parent
    # hyperparameters
    load_model = False
    model_path = current_dir / Path("model/model_1470000.pth")
    # ------------
    train_env = model_env.TrainEnv()
    train_env.setup_tensorboard(Path("/root/tf-logs/llm_fixed"))

    if load_model == False:
        with (
            current_dir / Path("../data_buffer/pretrain_t2t_mini/character_mapper.pkl")
        ).open("rb") as f:
            character_mapper: units.CharacterMapper = pickle.load(f)
        model = LLMFixedModel(
            character_mapper,
            token_embd=192,
            head_nums=4,
            head_size=64,
            n_layer=4,
            infer_vec_nums=32,
            infer_dim=384,
            forecast_steps=1,
        )
        train_env.set_model(model)
    else:
        train_env.load_model(model_path=model_path)
    train_env.model_summary(input_size=(1, 512))

    train_env.setup_optimizers(
        learning_rate=2e-4,
        warmup_iters=4096,
        cos_T_0=1024 * 16,
        cos_T_mult=2,
        cos_eta_min=1e-7,
    )

    train_env.load_data(
        current_dir / Path("../data_buffer/pretrain_t2t_mini/train_data.pt"),
        current_dir / Path("../data_buffer/pretrain_t2t_mini/val_data.pt"),
    )

    train_env.train_loop(
        save_iters=10000,
        eval_interval=1000,
        batch_size=512,
        eval_iters=10,
        max_data_len=256,
        save_dir=current_dir / Path("model"),
    )


if __name__ == "__main__":
    train()
