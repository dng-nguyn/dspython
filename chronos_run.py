import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np

# Load monthly tourism data
df = pd.read_csv('output/df_monthly.csv')
agg = df.groupby(['year', 'month'])['arrivals'].sum().reset_index()
agg['date'] = pd.to_datetime(agg['year'].astype(str) + '-' + agg['month'].astype(str) + '-01')
agg = agg.sort_values('date').reset_index(drop=True)

# Train-Test Split
train_ts = agg[(agg['year'] >= 2012) & (agg['year'] <= 2019)]['arrivals'].values
test_ts = agg[(agg['year'] >= 2022) & (agg['year'] <= 2025)]['arrivals'].values

print("Train TS length:", len(train_ts))
print("Test TS length:", len(test_ts))

context = torch.tensor(train_ts.astype(np.float32))

results = {}
for size in ["tiny", "small", "base"]:
    model_name = f"amazon/chronos-t5-{size}"
    print(f"Running {model_name}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    # Predict
    forecast = pipeline.predict(context, prediction_length=48, num_samples=100) # use 100 samples for better median estimation
    samples = forecast[0].numpy() # shape: (num_samples, 48)
    median_pred = np.median(samples, axis=0)
    
    # Calculate MAE
    mae = np.mean(np.abs(test_ts - median_pred))
    print(f"{model_name} MAE: {mae:,.2f}")
    results[size] = {
        'mae': mae,
        'preds': median_pred
    }

# Find best
best_size = min(results.keys(), key=lambda k: results[k]['mae'])
print(f"Best Chronos size: {best_size} with MAE {results[best_size]['mae']:,.2f}")

# Save the predictions of all sizes and the best one to a temporary file
np.savez('output/chronos_preds.npz', 
         best_size=best_size,
         tiny=results['tiny']['preds'],
         small=results['small']['preds'],
         base=results['base']['preds'])
print("Saved Chronos predictions to output/chronos_preds.npz")
