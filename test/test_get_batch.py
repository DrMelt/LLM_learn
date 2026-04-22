import os
import sys
from pathlib import Path
import json

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from shared.units import get_batch, CharacterMapper


def test_get_batch():
    # 构造一个 CharacterMapper，用于编码文本
    mapper = CharacterMapper(chars_in="abcdefghijklmnopqrstuvwxyz")

    # 构造训练数据：3 个长度不同的条目
    entries = [
        torch.tensor(mapper.encode("abcdefghij"), dtype=torch.long),  # 长度 10
        torch.tensor(mapper.encode("klmnopqrstuvwxyz"), dtype=torch.long),  # 长度 16
        torch.tensor(mapper.encode("abc"), dtype=torch.long),  # 长度 3（需要填充）
    ]

    block_size = 8
    batch_size = 4
    target_len = 4
    target_offset = 1
    pad_value = mapper.stoi[mapper.pad_token]

    print(f"pad_token: {mapper.pad_token!r}, pad_value: {pad_value}")
    print(
        f"unk_token:  {mapper.unk_token!r}, unk_value:  {mapper.stoi[mapper.unk_token]}"
    )
    print(f"entries lengths: {[e.size(0) for e in entries]}")
    print(
        f"block_size={block_size}, batch_size={batch_size}, target_len={target_len}, target_offset={target_offset}"
    )
    print(f"min_len={max(block_size, target_offset + target_len)}")
    print("-" * 60)

    x, y = get_batch(
        data=entries,
        block_size=block_size,
        batch_size=batch_size,
        target_len=target_len,
        target_offset=target_offset,
        pad_value=pad_value,
    )

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print("-" * 60)

    for i in range(batch_size):
        xi = x[i].tolist()
        yi = y[i].tolist()
        print(f"batch {i}:")
        print(f"  x (decoded): {mapper.decode(xi)!r}")
        print(f"  y (decoded): {mapper.decode(yi)!r}")

        # 高亮显示填充部分
        pad_count = sum(1 for v in xi if v == pad_value)
        if pad_count:
            print(f"  -> 包含 {pad_count} 个填充位 (PAD)")

        # 验证 y 是否是 x 的偏移
        for j, y_idx in enumerate(yi):
            expected = xi[target_offset + j]
            if y_idx != expected:
                print(f"  [WARN] y[{j}]={y_idx} != x[{target_offset + j}]={expected}")
        print()


def test_pretrain_t2t_mini():
    """测试 pretrain_t2t_mini 数据集的读取和采样（从 data_buffer 加载）。"""
    buffer_dir = project_root / "data_buffer" / "pretrain_t2t_mini"

    # 1. 加载已保存的 CharacterMapper 和数据
    import pickle

    with open(buffer_dir / "character_mapper.pkl", "rb") as f:
        mapper: CharacterMapper = pickle.load(f)
    print(f"词表大小: {len(mapper.chars)}")
    print(f"pad_token: {mapper.pad_token!r} -> index {mapper.stoi[mapper.pad_token]}")
    print(f"unk_token: {mapper.unk_token!r} -> index {mapper.stoi[mapper.unk_token]}")

    train_data: list[torch.Tensor] = torch.load(
        buffer_dir / "train_data.pt", weights_only=False
    )
    val_data: list[torch.Tensor] = torch.load(
        buffer_dir / "val_data.pt", weights_only=False
    )
    print(f"加载完成 — 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")

    print(
        f"训练集长度分布: min={min(e.size(0) for e in train_data)}, "
        f"max={max(e.size(0) for e in train_data)}, "
        f"avg={sum(e.size(0) for e in train_data) / len(train_data):.1f}"
    )
    print("-" * 60)

    # 2. 采样参数
    block_size = 64
    batch_size = 4
    target_len = 64
    target_offset = 1
    pad_value = mapper.stoi[mapper.pad_token]

    print(
        f"采样参数: block_size={block_size}, batch_size={batch_size}, target_len={target_len}, target_offset={target_offset}"
    )
    print(f"min_len={max(block_size, target_offset + target_len)}")
    print("-" * 60)

    x, y = get_batch(
        data=train_data,
        block_size=block_size,
        batch_size=batch_size,
        target_len=target_len,
        target_offset=target_offset,
        pad_value=pad_value,
    )

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print("-" * 60)

    for i in range(batch_size):
        xi = x[i].tolist()
        yi = y[i].tolist()
        pad_count = sum(1 for v in xi if v == pad_value)

        print(f"batch {i}: {'[含填充]' if pad_count else '[无填充]'}")
        print(f"  x 前 30 字符: {mapper.decode(xi[:30])!r}")
        print(f"  y 前 30 字符: {mapper.decode(yi[:30])!r}")
        if pad_count:
            print(f"  -> 包含 {pad_count} 个填充位")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 简单构造数据")
    print("=" * 60)
    test_get_batch()

    print("\n" + "=" * 60)
    print("测试 2: pretrain_t2t_mini 数据集")
    print("=" * 60)
    test_pretrain_t2t_mini()
