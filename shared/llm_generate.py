import os
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from shared.model_base import LLM_ModelBase

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: LLM_ModelBase = (
        torch.load(
            f"llm_fixed_model/llm_fixed_model_{26*10000}.pth", weights_only=False
        )
        .to(device)
        .eval()
    )
    character_mapper = model.vocab_map

    context = torch.tensor(
        character_mapper.encode("唐三你"),
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    open("more.txt", "w").write(
        character_mapper.decode(
            model.generate(context, max_new_tokens=2000, temperature=1.0)[0].tolist()
        )
    )
