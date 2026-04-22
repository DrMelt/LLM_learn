import torch
from torch import nn

from .units import CharacterMapper, get_batch


class LLM_ModelBase(nn.Module):
    def __init__(self, vocab_map: CharacterMapper, out_nums: int):
        super().__init__()
        self.iter_n: int = 0

        self.vocab_map: CharacterMapper = vocab_map
        self.out_nums: int = out_nums

    def train_step(
        self,
        data: list[torch.Tensor],
        max_data_len: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        raise NotImplementedError("train_step method must be implemented in subclass")

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        raise NotImplementedError("generate method must be implemented in subclass")

    def _target_offset_func(self, block_size: int) -> int:
        raise NotImplementedError(
            "target_offset_func method must be implemented in subclass"
        )

    @torch.no_grad()
    def estimate_loss(
        self,
        block_size_range: tuple[int, int],
        batch_size: int,
        eval_iters: int,
        train_data: list[torch.Tensor],
        val_data: list[torch.Tensor],
    ):
        out = {}
        self.eval()
        block_size_samples = torch.linspace(
            block_size_range[0],
            block_size_range[1],
            steps=eval_iters,
            dtype=torch.int32,
        ).tolist()
        data = {"train": train_data, "val": val_data}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(
                    data=data[split],
                    block_size=block_size_samples[k],
                    batch_size=batch_size,
                    target_len=self.out_nums,
                    target_offset=self._target_offset_func(block_size_samples[k]),
                )
                _, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
