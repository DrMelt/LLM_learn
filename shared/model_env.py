import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import pickle
from torch.optim.lr_scheduler import LRScheduler

from .units import CharacterMapper, get_batch
from .module import LLM_ModelBase


class TrainEnv:
    def __init__(
        self,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LLM_ModelBase] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.writer: Optional[SummaryWriter] = None
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None

    def set_model(
        self,
        model: LLM_ModelBase,
    ):
        self.model = model
        assert self.model is not None, "Model is not set"
        self.model.to(self.device).train()
        return self.model

    def load_model(self, model_path: Path) -> LLM_ModelBase:
        """从文件加载模型"""
        self.model = torch.load(model_path, weights_only=False)
        assert self.model is not None, "Model loading failed"
        self.model.to(self.device).train()
        return self.model

    def setup_optimizers(
        self,
        learning_rate: float,
        warmup_iters: int,
        cos_T_0: int,
        cos_T_mult: int,
        cos_eta_min: float,
    ):
        """设置优化器和学习率调度器"""
        assert self.model is not None, "Model is not set"

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 预热学习率调度器
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_iters,
        )

        # 余弦退火学习率调度器
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cos_T_0,
            T_mult=cos_T_mult,
            eta_min=cos_eta_min,
        )

        # 顺序组合学习率调度器
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_iters],
        )

        return self.optimizer, self.scheduler

    def load_data(self, train_data_path: Path, val_data_path: Path):
        """加载训练和验证数据"""
        self.train_data = torch.load(train_data_path)
        self.val_data = torch.load(val_data_path)
        return self.train_data, self.val_data

    def setup_tensorboard(self, log_dir: Path):
        """设置 TensorBoard 日志记录"""
        self.writer = SummaryWriter(log_dir)
        return self.writer

    def train_step(
        self,
        batch_size: int,
        max_data_len: int = 512,
    ):
        """执行单步训练"""
        assert self.model is not None, "Model is not set"
        assert self.train_data is not None, "Train data is not set"
        assert self.optimizer is not None, "Optimizer is not set"
        assert self.scheduler is not None, "Scheduler is not set"

        self.model.train_step(
            data=self.train_data,
            max_data_len=max_data_len,
            batch_size=batch_size,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def evaluate_model(
        self,
        batch_size: int,
        eval_iters: int,
        block_size_range: tuple,
    ) -> Dict[str, float]:
        """评估模型在训练集和验证集上的损失"""
        assert (
            self.model is not None
            and self.train_data is not None
            and self.val_data is not None
        ), "请先初始化模型和数据"

        losses = self.model.estimate_loss(
            block_size_range=block_size_range,
            batch_size=batch_size,
            eval_iters=eval_iters,
            train_data=self.train_data,
            val_data=self.val_data,
        )
        return losses

    def save_model(self, save_dir: Path = Path("model")):
        """保存模型检查点"""
        assert self.model is not None, "没有可保存的模型"

        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"model_{self.model.iter_n}.pth"
        torch.save(self.model, model_path)
        print(f"模型已保存到 {model_path}")

    def log_metrics(self, losses: Dict[str, float]):
        """记录训练指标到 TensorBoard"""
        assert self.writer is not None, "请先调用 setup_tensorboard"
        assert self.model is not None, "请先初始化模型"
        assert self.scheduler is not None, "请先初始化学习率调度器"

        self.writer.add_scalar(
            "Learning Rate", self.scheduler.get_last_lr()[0], self.model.iter_n
        )
        self.writer.add_scalar("Loss/train", losses["train"], self.model.iter_n)
        self.writer.add_scalar("Loss/val", losses["val"], self.model.iter_n)

    def print_training_status(self, losses: Dict[str, float]):
        """打印训练状态"""
        assert self.model is not None, "请先初始化模型"
        assert self.scheduler is not None, "请先初始化学习率调度器"

        lr = self.scheduler.get_last_lr()[0]
        lr_str = f"\t lr {lr:.2e}" if lr is not None else ""
        print(
            f"step {self.model.iter_n}\t{lr_str}\t train loss {losses['train']:.4e}\t val loss {losses['val']:.4e}"
        )

    def model_summary(self, input_size: tuple, depth: int = 3):
        """打印模型摘要信息"""
        assert self.model is not None, "请先初始化模型"

        summary(
            self.model,
            input_size=input_size,
            dtypes=[torch.long],
            row_settings=["var_names"],
            depth=depth,
        )

    def train_loop(
        self,
        save_iters: int,
        eval_interval: int,
        batch_size: int,
        eval_iters: int,
        max_data_len: int,
        save_dir: Path,
    ):
        assert self.model is not None, "请先初始化模型"

        while True:
            if self.model.iter_n % save_iters == 0:
                self.save_model(save_dir=save_dir)

            if self.model.iter_n % eval_interval == 0:
                losses = self.evaluate_model(
                    batch_size=batch_size,
                    eval_iters=eval_iters,
                    block_size_range=(1, max_data_len),
                )
                self.print_training_status(
                    losses=losses,
                )
                self.log_metrics(
                    losses=losses,
                )

            self.train_step(batch_size=batch_size, max_data_len=max_data_len)
