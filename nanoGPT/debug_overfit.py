import os
from pathlib import Path
import torch

from llm_fixed import LLMFixedModel
from units import CharacterMapper, get_batch, read_file

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    torch.manual_seed(42)
    block_size = 1024
    batch_size = 16

    # pass a Path object to match the signature of read_files
    text = read_file(Path("../inputs/inputs_cn/test.txt"))  # 只取训练文本
    chars = sorted(list(set(text)))
    character_mapper = CharacterMapper(chars_in=chars)

    model = (
        LLMFixedModel(
            character_mapper,
            token_embd=32,
            head_nums=2,
            head_size=16,
            n_layer=2,
            infer_vec_nums=16,
            infer_dim=64,
        )
        .to(device)
        .train()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = torch.tensor(character_mapper.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    xb, yb = get_batch(train_data, block_size, batch_size, is_single_target=True)

    for i in range(200):
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i % 10 == 0:
            print(f"step {i} loss {loss.item():.6f}")

    print("overfit test finished")


if __name__ == "__main__":
    main()
