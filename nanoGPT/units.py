import os
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


# select cuda if available otherwise fall back to cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharacterMapper:
    def __init__(self, chars_in: Iterable[str], max_vocab_size: int = 32768):
        self.max_vocab_size = max_vocab_size
        # 确保 <UNK> 在索引 0 位置，并且只遍历一次、去掉重复项
        self.unk_token = "<UNK>"
        seen = {self.unk_token}
        self.chars = [self.unk_token]
        for c in chars_in:
            if len(self.chars) >= self.max_vocab_size:
                # 达到上限，不再添加新的字符
                break
            if c in seen:  # 跳过已经见过的字符
                continue
            seen.add(c)
            self.chars.append(c)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def _encode_char(self, ch: str) -> int:
        """把新的字符加到表尾并返回其索引。达到 max_vocab_size 时不再添加，返回 <UNK> 的索引。"""
        if ch in self.stoi:
            # 已经在表里了，直接返回索引
            return self.stoi[ch]
        if len(self.chars) >= self.max_vocab_size:
            # 超过容量，全部映射为 unk
            return self.stoi[self.unk_token]
        idx = len(self.chars)
        self.chars.append(ch)
        self.stoi[ch] = idx
        self.itos[idx] = ch
        return idx

    def encode(self, string: str) -> list[int]:
        """把字符串转成索引列表。
        遇到表外字符时把该字符加进表（如果尚未达到上限），
        否则映射为 <UNK>。
        """
        res: list[int] = []
        for c in string:
            res.append(self._encode_char(c))
        return res

    def decode(self, str_code: list[int]) -> str:
        """把索引列表转回字符串。索引出界的项会被替换为 <UNK>。"""
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
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    target_len: int,
    target_offset: int,
):

    # 允许的起始位置个数，要保证 x 窗口和 y 窗口都不溢出
    max_start_idx = data.size(0) - max(block_size, target_offset + target_len) + 1
    if max_start_idx <= 0:
        raise ValueError("数据长度不足以生成指定大小的窗口")
    ix = torch.randint(max_start_idx, (batch_size,))
    full_x = data.as_strided(size=(max_start_idx, block_size), stride=(1, 1))
    x = full_x[ix]  # (batch_size, block_size)
    full_y = data.as_strided(
        size=(max_start_idx, target_len),
        stride=(1, 1),
        storage_offset=target_offset,
    )

    y = full_y[ix]  # (batch_size, target_len)
    return x.to(device), y.to(device)




if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    text_train = read_file(Path("../inputs/inputs_cn/text_train_cn.txt"))
    text_val = read_file(Path("../inputs/inputs_cn/text_val_cn.txt"))
    character_mapper = CharacterMapper(chars_in=set(text_train + text_val))

    # 立刻用 mapper 把文本转成张量并把这些对象存盘
    train_data = torch.tensor(character_mapper.encode(text_train), dtype=torch.long)
    val_data = torch.tensor(character_mapper.encode(text_val), dtype=torch.long)
    buffer_dir = Path("buffer")
    buffer_dir.mkdir(exist_ok=True)

    with open(buffer_dir / "character_mapper.pkl", "wb") as f:
        pickle.dump(character_mapper, f)

    torch.save(train_data, buffer_dir / "train_data.pt")
    torch.save(val_data, buffer_dir / "val_data.pt")
