import os
import sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, record_function


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from shared.units import *
from shared.module import *

from llm_fixed.llm_fixed import *

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: LLMFixedModel = (
        torch.load(current_dir / f"model/model_{100*10000}.pth", weights_only=False)
        .to(device)
        .eval()
    )
    character_mapper = model.vocab_map

    context = torch.tensor(
        character_mapper.encode(
            "换作平时，八寻会等男人醒来后再郑重道歉，但目前时间紧迫"
        ),
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    open(current_dir / "forecast.txt", "w").write(
        character_mapper.decode(
            model.generate(context, max_new_tokens=2000, temperature=1.0)[0].tolist()
        )
    )
