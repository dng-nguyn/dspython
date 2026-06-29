import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np

print("Loading pipeline...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)
print("Pipeline loaded successfully!")
