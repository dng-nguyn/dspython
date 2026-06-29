import torch
from chronos import ChronosPipeline

for size in ["tiny", "small", "base"]:
    model_name = f"amazon/chronos-t5-{size}"
    print(f"Loading {model_name}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"Loaded {model_name} successfully!")
