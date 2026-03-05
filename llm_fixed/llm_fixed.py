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


class LLMFixedBlock(nn.Module):
    def __init__(self, a_embd: int, b_embd: int, head_nums: int, head_size: int):
        super().__init__()
        self.block_a2b = Block_A2B(
            a_embd=a_embd,
            b_embd=b_embd,
            n_head=head_nums // 2,
            head_size=head_size // 2,
        )
        self.block_self = Block_Self(
            n_embd=b_embd, n_head=head_nums, head_size=head_size
        )
        self.adaptive_norm = AdaptiveVectorModifier(
            vector_dim=b_embd, feature_dim=b_embd // 8
        )

    def forward(self, a, b):
        b = self.block_a2b(a, b)
        b = self.block_self(b)
        b = self.adaptive_norm(b)
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


class LLMFixedVec2Word(nn.Module):
    def __init__(
        self, infer_n_embd: int, projection_dim: int, vocab_size: int, out_nums: int
    ):
        super().__init__()
        self.out_nums = out_nums
        self.projection = nn.Linear(infer_n_embd, projection_dim, bias=False)
        self.vec_to_word = nn.Linear(projection_dim, vocab_size, bias=False)

    def forward(self, inference_vec):
        # only use the first of the infer_vec_size dimension
        first_vec = inference_vec[:, : self.out_nums, :]  # (B, out_nums, infer_n_embd)
        first_vec = self.projection(first_vec)  # (B, out_nums, projection_dim)
        logits = self.vec_to_word(first_vec)  # (B, out_nums, vocab_size)
        return logits


class LLMFixedModel(LLM_ModelBase):
    def __init__(
        self,
        vocab_map: CharacterMapper,
        token_embd: int,
        head_nums: int,
        head_size: int,
        n_layer: int,
        infer_vec_nums: int,
        infer_dim: int,
        out_nums: int,
    ):
        super().__init__(
            vocab_map=vocab_map,
            out_nums=out_nums,
        )
        self.iter_n = 0

        self.out_nums = out_nums
        self.token_embd = token_embd
        self.info_n_embd = infer_dim
        self.infer_vec_nums = infer_vec_nums
        self.register_buffer("infer_arange", torch.arange(infer_vec_nums))
        self.infer_vec_embedder = nn.Embedding(infer_vec_nums, infer_dim)
        # each token directly reads off the logits for the next token from a lookup table
        self.embedder = CharacterEmbedder(
            vocab_size=vocab_map.max_vocab_size, n_embd=token_embd
        )

        self.blocks = LLMFixedBlocks(
            n_layer=n_layer,
            head_nums=head_nums,
            head_size=head_size,
            a_embd=token_embd,
            b_embd=infer_dim,
        )

        self.vec_to_word = LLMFixedVec2Word(
            infer_n_embd=infer_dim,
            projection_dim=token_embd,
            vocab_size=vocab_map.max_vocab_size,
            out_nums=out_nums,
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

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        inference_vec = self.infer_vec_embedder(
            self.get_buffer("infer_arange")
        )  # (info_vec_size, info_n_embd)
        inference_vec = inference_vec.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, info_vec_size, info_n_embd)
        x = self.embedder.embed(idx)  # (B,T,C)

        inference_vec = self.blocks(x, inference_vec)  # (B, info_vec_size, info_n_embd)
        logits = self.vec_to_word(inference_vec)  # (B, vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * self.out_nums, -1)
            targets = targets.view(-1)
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
        data: torch.Tensor,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        data_sample_len = random.randint(1, max_data_len)
        xb, yb = get_batch(
            data=data,
            block_size=data_sample_len,
            batch_size=batch_size,
            target_len=self.out_nums,
            target_offset=data_sample_len,
        )
        logits, loss = self(xb, yb)
        loss = loss * (data_sample_len / (data_sample_len + 8.0))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()  # 更新学习率
        self.iter_n += 1


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # hyperparameters
    max_data_len = 512
    batch_size = 256
    learning_rate = 2e-4
    warmup_iters = 5000
    save_iters = 10000
    eval_interval = 1000
    eval_iters = 10

    out_nums = 4

    load_model = False
    model_path = "llm_fixed_model/llm_fixed_model_260000.pth"
    # ------------
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    writer = SummaryWriter("/root/tf-logs/llm_fixed")

    if load_model == False:
        with open("buffer/character_mapper.pkl", "rb") as f:
            character_mapper = pickle.load(f)
        model = LLMFixedModel(
            character_mapper,
            token_embd=64,
            head_nums=8,
            head_size=64,
            n_layer=8,
            infer_vec_nums=64,
            infer_dim=256,
            out_nums=out_nums,
        )
    else:
        model: LLMFixedModel = torch.load(model_path, weights_only=False)
        character_mapper = model.vocab_map

    model.to(device).train()
    summary(
        model,
        input_size=(1, max_data_len),
        dtypes=[torch.long],
        row_settings=["var_names"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=4096,  # 第一次重启前的迭代次数
        T_mult=2,  # 重启后周期倍增
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cos_scheduler],
        milestones=[warmup_iters],
    )

    train_data = torch.load("buffer/train_data.pt")
    val_data = torch.load("buffer/val_data.pt")

    while True:
        if model.iter_n % save_iters == 0:
            torch.save(model, f"llm_fixed_model/llm_fixed_model_{model.iter_n}.pth")

        if model.iter_n % eval_interval == 0:
            losses = model.estimate_loss(
                block_size_range=(1, max_data_len),
                batch_size=batch_size,
                eval_iters=eval_iters,
                train_data=train_data,
                val_data=val_data,
                target_offset_func=lambda x: x,
            )

            print(
                f"step {model.iter_n}\t lr {scheduler.get_last_lr()[0]:.2e}\t train loss {losses['train']:.4e}\t val loss {losses['val']:.4e}"
            )
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], model.iter_n)
            writer.add_scalar("Loss/train", losses["train"], model.iter_n)
            writer.add_scalar("Loss/val", losses["val"], model.iter_n)

        model.train_step(
            data=train_data,
            batch_size=batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )
