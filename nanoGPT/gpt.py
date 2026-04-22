import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from shared.units import *
from shared.module import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_map: CharacterMapper,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.mapper = vocab_map
        # each token directly reads off the logits for the next token from a lookup table
        self.embedder = CharacterEmbedder(
            vocab_size=vocab_map.max_vocab_size, n_embd=n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(n_embd=n_embd, n_head=n_head, block_size=block_size)
                for _ in range(n_layer)
            ]
        )
        # self.layer_norm = nn.LayerNorm(n_embd)  # final layer norm
        self.vec_to_word = nn.Linear(n_embd, vocab_map.max_vocab_size)

        self.register_buffer("loss_weight", torch.arange(block_size) / block_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        x = self.embedder(idx)  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        # x = self.layer_norm(x)  # (B,T,C)
        logits = self.vec_to_word(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, reduction="none").view(B, T)
            loss = loss * self.get_buffer("loss_weight")[:T]
            loss = loss.mean()

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    # hyperparameters
    block_size = 256  # what is the maximum context length for predictions?
    batch_size = 128  # how many independent sequences will we process in parallel?
    learning_rate = 1e-4
    save_iters = 10000
    eval_interval = 500
    eval_iters = 10
    # ------------
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    writer = SummaryWriter("/root/tf-logs/nanoGPT")

    text_train, text_val = read_files(Path("inputs"), train_p=0.9)
    # here are all the unique characters that occur in this text

    chars = sorted(list(set(text_train + text_val)))
    character_mapper = CharacterMapper(chars_in=chars)

    print("vocab size:", character_mapper.max_vocab_size)
    # model = GPTLanguageModel(
    #     character_mapper,
    #     n_embd=64,
    #     n_head=4,
    #     n_layer=4,
    #     block_size=block_size,
    # ).to(device)
    model = torch.load("model.pth", weights_only=False).to(device)

    # Train and test splits（按条目封装为 list[torch.Tensor]）
    train_data = [
        torch.tensor(
            character_mapper.encode(text_train), dtype=torch.long, device=device
        )
    ]
    val_data = [
        torch.tensor(character_mapper.encode(text_val), dtype=torch.long, device=device)
    ]

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    iter_n = 0
    while True:
        for _ in range(save_iters):
            iter_n += 1
            # every once in a while evaluate the loss on train and val sets
            if iter_n % eval_interval == 0 or iter_n == save_iters - 1:
                losses = model.estimate_loss(
                    block_size_range=(block_size, block_size),
                    batch_size=batch_size,
                    eval_iters=eval_iters,
                    train_data=train_data,
                    val_data=val_data,
                    target_offset_func=lambda x: 1,
                )

                print(
                    f"step {iter_n}:\t train loss {losses['train']:.4f},\t val loss {losses['val']:.4f}"
                )
                writer.add_scalar("Loss/train", losses["train"], iter_n)
                writer.add_scalar("Loss/val", losses["val"], iter_n)

            # sample a batch of data
            xb, yb = get_batch(
                data=train_data,
                block_size=block_size,
                batch_size=batch_size,
                target_len=block_size,
                target_offset=1,
            )

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, "model.pth")
