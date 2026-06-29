import torch
from chronos import ChronosPipeline
import numpy as np

# Create dummy context of 96 points
context = torch.tensor(np.random.randn(96).astype(np.float32))

print("Loading pipeline...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)

print("Predicting...")
forecast = pipeline.predict(context, prediction_length=48)
print("Forecast shape:", forecast.shape)
print("Forecast type:", type(forecast))
