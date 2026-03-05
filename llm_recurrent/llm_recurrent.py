import os
from pathlib import Path
import random
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import pickle

from shared.units import *
from shared.module import *

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    pass
