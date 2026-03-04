import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from gpt import *
from llm_fixed import *


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

    # inputs = torch.zeros(512, 1024, device=device, dtype=torch.long)
    # model(inputs)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    # ) as prof:
    #     with record_function("model_inference"):
    #         model(inputs)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    open("more.txt", "w").write(
        character_mapper.decode(
            model.generate(context, max_new_tokens=2000, temperature=1.0)[0].tolist()
        )
    )
