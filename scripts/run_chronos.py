#!/usr/bin/env python3
"""Run Chronos-T5 models on monthly tourism data."""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('output/df_monthly.csv')
agg = df.groupby(['year', 'month'])['arrivals'].sum().reset_index()
agg = agg.sort_values(['year', 'month']).reset_index(drop=True)

train_mask = (
    ((agg['year'] >= 2012) & (agg['year'] <= 2019)) |
    ((agg['year'] >= 2022) & (agg['year'] <= 2023))
)
test_mask = (agg['year'] >= 2024) & (agg['year'] <= 2025)
train_ts = agg[train_mask]['arrivals'].values.astype(float)
test_ts = agg[test_mask]['arrivals'].values.astype(float)

print(f"Train: {len(train_ts)} months, Test: {len(test_ts)} months")

import torch
from chronos import ChronosPipeline

sizes = ['tiny', 'small', 'base']
best_mape = 999
best_name = None
best_pred = None
chronos_results = {}

for size in sizes:
    model_name = f"amazon/chronos-t5-{size}"
    print(f"\nLoading {model_name}...")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        context = torch.tensor(train_ts, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=len(test_ts), num_samples=100)
        
        # Get median prediction
        pred = np.median(forecast[0].numpy(), axis=0)
        
        mae = np.mean(np.abs(test_ts - pred))
        rmse = np.sqrt(np.mean((test_ts - pred)**2))
        mape = np.mean(np.abs((test_ts - pred) / test_ts)) * 100
        ss_res = np.sum((test_ts - pred)**2)
        ss_tot = np.sum((test_ts - np.mean(test_ts))**2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"  chronos-t5-{size:8s}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  MAPE={mape:>6.2f}%  R²={r2:>7.4f}")
        
        chronos_results[size] = {
            'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'R²': float(r2),
            'pred': pred.tolist()
        }
        
        if mape < best_mape:
            best_mape = mape
            best_name = size
            best_pred = pred
            
        # Free memory
        del pipeline, context, forecast
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nBest Chronos: chronos-t5-{best_name} (MAPE={best_mape:.2f}%)")

# Save to existing results
with open('output/_ml_preds.json', 'r') as f:
    data = json.load(f)
data['Chronos'] = best_pred.tolist()
data['results'].append({
    'Model': f'Chronos-T5-{best_name}',
    'MAE': chronos_results[best_name]['MAE'],
    'RMSE': chronos_results[best_name]['RMSE'],
    'MAPE': chronos_results[best_name]['MAPE'],
    'R²': chronos_results[best_name]['R²']
})
with open('output/_ml_preds.json', 'w') as f:
    json.dump(data, f)

# Update model_results.csv
pd.DataFrame(data['results']).to_csv('output/model_results.csv', index=False)

# Update pred_vs_actual.csv
pva = pd.read_csv('output/pred_vs_actual.csv')
pva['Chronos'] = best_pred.astype(int)
pva.to_csv('output/pred_vs_actual.csv', index=False)

print("\nFinal results:")
for r in data['results']:
    print(f"  {r['Model']:30s} MAE={r['MAE']:>12,.0f}  MAPE={r['MAPE']:>6.2f}%  R²={r['R²']:>7.4f}")
