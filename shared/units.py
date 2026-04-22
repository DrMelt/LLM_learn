import os
import random
from pathlib import Path
import chardet
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable
import pickle
import json


# select cuda if available otherwise fall back to cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharacterMapper:
    """字符 ↔ 索引映射器。

    词表结构（前 ``reserved_slots`` 个位置固定预留）：
    - 索引 0 : ``<PAD>``   — 填充符号
    - 索引 1 : ``<UNK>``   — 未知字符
    - 索引 2 … ``reserved_slots-1`` : ``<RESERVED_n>`` — 占位符，供后续扩展
    - 索引 ``reserved_slots`` … : 实际字符
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def __init__(
        self,
        chars_in: Iterable[str],
        max_vocab_size: int = 32768,
        reserved_slots: int = 32,
    ):
        self.max_vocab_size = max_vocab_size
        self.reserved_slots = reserved_slots

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.chars = self._build_vocab(chars_in)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def _build_vocab(self, chars_in: Iterable[str]) -> list[str]:
        """构建词表：先插入预留符号，再依次加入实际字符。"""
        seen: set[str] = set()
        chars: list[str] = []

        # 1. 固定预留符号
        fixed_tokens = (self.pad_token, self.unk_token)
        for token in fixed_tokens:
            seen.add(token)
            chars.append(token)

        # 2. 可扩展占位符（从 fixed_tokens 之后开始）
        for i in range(len(fixed_tokens), self.reserved_slots):
            token = f"<RESERVED_{i}>"
            seen.add(token)
            chars.append(token)

        # 3. 实际字符
        for c in chars_in:
            if len(chars) >= self.max_vocab_size:
                break
            if c in seen:
                continue
            seen.add(c)
            chars.append(c)

        return chars

    # ------------------------------------------------------------------
    # 编解码
    # ------------------------------------------------------------------
    def encode(self, string: str) -> list[int]:
        """把字符串转成索引列表。
        遇到表外字符时动态加入词表（未达上限），否则映射为 ``<UNK>``。
        """
        return [self._encode_char(c) for c in string]

    def _encode_char(self, ch: str) -> int:
        """单字符编码。若字符不在词表中且未达上限，则动态追加。"""
        if ch in self.stoi:
            return self.stoi[ch]
        if len(self.chars) >= self.max_vocab_size:
            return self.stoi[self.unk_token]

        idx = len(self.chars)
        self.chars.append(ch)
        self.stoi[ch] = idx
        self.itos[idx] = ch
        return idx

    def decode(self, str_code: list[int]) -> str:
        """把索引列表转回字符串。索引出界的项替换为 ``<UNK>``。"""
        return "".join(self.itos.get(i, self.unk_token) for i in str_code)


def read_epub_content(book: epub.EpubBook):
    content = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                # 使用 BeautifulSoup 解析 HTML 内容
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                # 提取纯文本内容
                content += soup.get_text()
                content += "\n"

            except UnicodeDecodeError:
                print(f"无法解码文件 {item.file_name}，跳过该文件")
    return content


def detect_encoding(file_path: Path, nbytes: int = 8192):
    # 只读前几个 KB 来猜测编码，chardet 对整个大文件没有必要
    with open(file_path, "rb") as f:
        raw = f.read(nbytes)
    return chardet.detect(raw).get("encoding", "utf-8")


def read_file(file_path: Path) -> str:
    try:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            enc = detect_encoding(file_path)
            with open(file_path, "r", encoding=enc, errors="replace") as infile:
                return infile.read()

        elif ext == ".epub":
            book = epub.read_epub(file_path)
            return read_epub_content(book)

    except Exception as e:
        print(f"无法读取文件 {file_path}，错误: {e}")
    return ""


def read_files(folder_path: Path, train_p: float = 0.9):
    train_chunks: list[str] = []
    val_chunks: list[str] = []
    paths: list[Path] = []
    for root, _, files in os.walk(folder_path):
        for fn in files:
            paths.append(Path(root) / fn)

    # 并行读取文件，IO 密集型任务用线程池即可

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(read_file, p): p for p in paths}
        for fut in as_completed(futures):
            content = fut.result()
            if not content:
                continue
            split_idx = int(len(content) * train_p)
            train_chunks.append(content[:split_idx])
            val_chunks.append(content[split_idx:])
    sep = "<file_split/>"
    return sep.join(train_chunks), sep.join(val_chunks)


def get_batch(
    data: list[torch.Tensor],
    block_size: int,
    batch_size: int,
    target_len: int,
    target_offset: int,
    pad_value: int = 0,
):
    """条目级采样：从 list[torch.Tensor] 中随机选条目，再在条目内随机位置采样。
    若条目长度不足，则用 ``pad_value`` 填充至所需长度。

    采用向量化批量采样：预分配 GPU 张量，减少 Python 循环开销与 CPU-GPU 同步。
    """
    entries: list[torch.Tensor] = data
    min_len = max(block_size, target_offset + target_len)
    num_entries = len(entries)

    # 预分配目标张量，避免后续 stack + to(device) 的二次拷贝
    x = torch.full((batch_size, block_size), pad_value, dtype=torch.long)
    y = torch.full((batch_size, target_len), pad_value, dtype=torch.long)

    for i in range(batch_size):
        # 向量化随机选条目，避免 random.choice 的 Python 开销
        idx = int(torch.randint(num_entries, (1,)).item())
        entry = entries[idx]
        entry_len = entry.size(0)

        if entry_len >= min_len:
            # 长度足够，随机采样起始位置
            max_start_idx = entry_len - min_len + 1
            start = int(torch.randint(max_start_idx, (1,)).item())
            x[i] = entry[start : start + block_size]
            y[i] = entry[start + target_offset : start + target_offset + target_len]
        else:
            # 长度不足，在开头用 pad_value 填充
            pad_len = min_len - entry_len
            padded = torch.cat(
                [
                    torch.full((pad_len,), pad_value, dtype=entry.dtype),
                    entry,
                ]
            )
            x[i] = padded[:block_size]
            y[i] = padded[target_offset : target_offset + target_len]

    return x.to(device), y.to(device)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)

    # 读取 pretrain_t2t_mini.jsonl，保留每个条目独立
    jsonl_path = Path("../inputs/pretrain_t2t_mini.jsonl")
    all_texts: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            all_texts.append(obj.get("text", ""))

    # 按条目数切分训练集和验证集
    split_idx = int(len(all_texts) * 0.9)
    text_train_entries = all_texts[:split_idx]
    text_val_entries = all_texts[split_idx:]

    # 收集所有字符构建词表
    all_chars = set()
    for text in all_texts:
        all_chars.update(text)
    character_mapper = CharacterMapper(chars_in=all_chars)

    # 每个条目独立编码为张量，保持列表结构
    train_data = [
        torch.tensor(character_mapper.encode(text), dtype=torch.long)
        for text in text_train_entries
    ]
    val_data = [
        torch.tensor(character_mapper.encode(text), dtype=torch.long)
        for text in text_val_entries
    ]

    buffer_dir = Path("../data_buffer/pretrain_t2t_mini")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    with open(buffer_dir / "character_mapper.pkl", "wb") as f:
        pickle.dump(character_mapper, f)

    torch.save(train_data, buffer_dir / "train_data.pt")
    torch.save(val_data, buffer_dir / "val_data.pt")
