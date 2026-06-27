#!/usr/bin/env python3
"""
Vietnam International Tourism Arrivals Analysis
Data: Quarterly arrivals by country (Q1-Q4), 2009-2026
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import html
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')

# ============================================================
# SECTION 1: DATA PARSING
# ============================================================
print("=" * 60)
print("SECTION 1: DATA PARSING")
print("=" * 60)

def parse_quarterly_file(filepath, quarter_label):
    """Parse HTML-Excel .xls file with Vietnamese tourism data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    tree = html.fromstring(raw)
    rows = tree.xpath('//tr')
    
    header_tds = rows[0].xpath('.//td')
    years = [(td.tail or '').strip() for td in header_tds[2:]]
    
    records = []
    for row in rows[1:]:
        tds = row.xpath('.//td')
        if len(tds) < 3:
            continue
        country = (tds[0].tail or '').strip()
        if not country or country == 'Phân theo thị trường':
            continue
        for j, td in enumerate(tds[1:]):
            val_str = (td.tail or '').strip()
            year_label = years[j] if j < len(years) else None
            if not val_str or not year_label or year_label == 'Totals':
                continue
            try:
                val = float(val_str.replace('.', ''))
            except ValueError:
                continue
            records.append({
                'country': country,
                'year': int(year_label),
                'quarter': quarter_label,
                'arrivals': val
            })
    return pd.DataFrame(records)

files = [
    ('Q1', 'data/quy1-cacnuoc.xls'),
    ('Q2', 'data/quy2-cacnuoc.xls'),
    ('Q3', 'data/quy3-cacnuoc.xls'),
    ('Q4', 'data/quy4-cacnuoc.xls'),
]

dfs = []
for label, fpath in files:
    df = parse_quarterly_file(fpath, label)
    dfs.append(df)
    print(f"  {label}: {len(df)} records, {df['country'].nunique()} countries, "
          f"years {df['year'].min()}-{df['year'].max()}")

df_long = pd.concat(dfs, ignore_index=True)

# Verification
hk_check = df_long[(df_long['country'] == 'Hoa Kỳ') & 
                    (df_long['year'] == 2009) & (df_long['quarter'] == 'Q1')]
assert hk_check['arrivals'].values[0] == 104520.0, "Verification failed!"
print(f"\n✓ Hoa Kỳ 2009 Q1 = {hk_check['arrivals'].values[0]:,.0f}")
print(f"df_long shape: {df_long.shape}")

# ============================================================
# SECTION 2: DATA CLEANING & COMBINING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: DATA CLEANING")
print("=" * 60)

# Filter out the 'Totals' row (it's a summary, not a country)
df_long = df_long[df_long['country'] != 'Totals'].reset_index(drop=True)

# Annual totals
df_total = df_long.groupby(['country', 'year'])['arrivals'].sum().reset_index()
df_total.columns = ['country', 'year', 'total_arrivals']
df_total = df_total.sort_values(['country', 'year']).reset_index(drop=True)

# Handle missing: fill with 0 for complete grid
df_long['arrivals'] = df_long['arrivals'].fillna(0)
df_total['total_arrivals'] = df_total['total_arrivals'].fillna(0)

all_countries = sorted(df_long['country'].unique())
all_years = sorted(df_long['year'].unique())
all_quarters = ['Q1', 'Q2', 'Q3', 'Q4']

idx = pd.MultiIndex.from_product([all_countries, all_years, all_quarters],
                                  names=['country', 'year', 'quarter'])
df_complete = df_long.set_index(['country', 'year', 'quarter']).reindex(idx, fill_value=0).reset_index()

print(f"df_complete shape: {df_complete.shape}")
print(f"Countries: {len(all_countries)}, Years: {len(all_years)}")
print(f"2021 data present: {2021 in df_long['year'].values}")

# Data quality
print(f"\nTop 10 by total arrivals:")
top10 = df_total.groupby('country')['total_arrivals'].sum().sort_values(ascending=False).head(10)
for c, v in top10.items():
    print(f"  {c}: {v/1e6:.2f}M")

print(f"\nYearly totals:")
yearly = df_total.groupby('year')['total_arrivals'].sum()
for y, v in yearly.items():
    print(f"  {y}: {v/1e6:.2f}M")

# ============================================================
# SECTION 3: FEATURE ENGINEERING & TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: FEATURE ENGINEERING")
print("=" * 60)

quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
df_complete['quarter_num'] = df_complete['quarter'].map(quarter_map)
df_complete['time_idx'] = df_complete['year'] + (df_complete['quarter_num'] - 1) / 4

df_complete = df_complete.sort_values(['country', 'year', 'quarter_num'])
for lag in [1, 2, 4]:
    df_complete[f'lag_{lag}'] = df_complete.groupby('country')['arrivals'].shift(lag)

df_complete['rolling_mean_4'] = (df_complete.groupby('country')['arrivals']
                                  .transform(lambda x: x.rolling(4, min_periods=1).mean()))

print(f"Features added: lag_1, lag_2, lag_4, rolling_mean_4")
print(f"df_complete shape: {df_complete.shape}")

# Train-test split
TRAIN_END = 2023
TEST_START = 2024

df_train = df_complete[df_complete['year'] <= TRAIN_END].copy()
df_test = df_complete[df_complete['year'] >= TEST_START].copy()
print(f"\nTrain: {df_train.shape[0]} rows (up to {TRAIN_END})")
print(f"Test:  {df_test.shape[0]} rows ({TEST_START}+)")

# ============================================================
# SECTION 4: EDA PLOTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Plot 1: Total arrivals trend
fig, ax = plt.subplots(figsize=(14, 7))
yearly_total = df_total.groupby('year')['total_arrivals'].sum().reset_index()
ax.plot(yearly_total['year'], yearly_total['total_arrivals'] / 1e6,
        'o-', linewidth=2, markersize=8, color='#2196F3')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Arrivals (millions)', fontsize=12)
ax.set_title('Vietnam International Tourist Arrivals by Year', fontsize=14, fontweight='bold')
ax.set_xticks(yearly_total['year'])
ax.set_xticklabels(yearly_total['year'], rotation=45)
ax.grid(True, alpha=0.3)
covid_val = yearly_total[yearly_total['year'] == 2020]['total_arrivals'].values[0] / 1e6
ax.annotate('COVID-19', xy=(2020, covid_val), xytext=(2018, covid_val + 1.5),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            color='red', fontweight='bold')
plt.tight_layout()
plt.savefig('output/eda_total_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_total_trend.png")

# Plot 2: Top 10 countries
top10_countries = (df_total.groupby('country')['total_arrivals'].sum()
                   .sort_values(ascending=False).head(10))
fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette('viridis', len(top10_countries))
bars = ax.barh(range(len(top10_countries)), top10_countries.values / 1e6, color=colors)
ax.set_yticks(range(len(top10_countries)))
ax.set_yticklabels(top10_countries.index, fontsize=11)
ax.set_xlabel('Total Arrivals (millions)', fontsize=12)
ax.set_title('Top 10 Source Countries — Total Arrivals', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, top10_countries.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val/1e6:.1f}M', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('output/eda_top10_countries.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_top10_countries.png")

# Plot 3: Seasonality
fig, ax = plt.subplots(figsize=(10, 6))
quarterly = df_long.groupby('quarter')['arrivals'].agg(['mean', 'std']).reindex(['Q1', 'Q2', 'Q3', 'Q4'])
ax.bar(quarterly.index, quarterly['mean'] / 1e3, yerr=quarterly['std'] / 1e3,
       capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], edgecolor='black')
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Mean Arrivals (thousands)', fontsize=12)
ax.set_title('Seasonality: Average Arrivals by Quarter', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output/eda_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_seasonality.png")

# Plot 4: Correlation heatmap (top 5)
top5 = top10_countries.head(5).index.tolist()
pivot = df_total[df_total['country'].isin(top5)].pivot_table(
    index='year', columns='country', values='total_arrivals', aggfunc='sum')
corr = pivot.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Correlation Between Top 5 Source Countries', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_correlation.png")

# Plot 5: Country-specific trends (top 5)
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()
for i, country in enumerate(top5):
    cdata = df_total[df_total['country'] == country].sort_values('year')
    axes[i].plot(cdata['year'], cdata['total_arrivals'] / 1e3, 'o-', linewidth=2, markersize=6)
    axes[i].set_title(country, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Arrivals (thousands)')
    axes[i].grid(True, alpha=0.3)
axes[5].set_visible(False)
plt.suptitle('Country-Specific Arrival Trends', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/eda_country_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_country_trends.png")

# ============================================================
# SECTION 5: MODELING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: MODELING")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Prepare features for ML models (aggregate: total arrivals per year-quarter)
agg_train = df_complete[df_complete['year'] <= TRAIN_END].groupby(
    ['year', 'quarter_num', 'time_idx'])['arrivals'].sum().reset_index()
agg_test = df_complete[df_complete['year'] >= TEST_START].groupby(
    ['year', 'quarter_num', 'time_idx'])['arrivals'].sum().reset_index()

# Add lag features for aggregate
agg_all = pd.concat([agg_train, agg_test]).sort_values(['year', 'quarter_num']).reset_index(drop=True)
agg_all['lag_1'] = agg_all['arrivals'].shift(1)
agg_all['lag_4'] = agg_all['arrivals'].shift(4)
agg_all['rolling_mean_4'] = agg_all['arrivals'].rolling(4, min_periods=1).mean()

# Drop rows with NaN lags for training
feature_cols = ['year', 'quarter_num', 'time_idx', 'lag_1', 'lag_4', 'rolling_mean_4']
agg_train_feat = agg_all[agg_all['year'] <= TRAIN_END].dropna(subset=['lag_1', 'lag_4']).copy()
agg_test_feat = agg_all[agg_all['year'] >= TEST_START].dropna(subset=['lag_1', 'lag_4']).copy()

X_train = agg_train_feat[feature_cols].values
y_train = agg_train_feat['arrivals'].values
X_test = agg_test_feat[feature_cols].values
y_test = agg_test_feat['arrivals'].values

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression:  MAE={mae_lr:,.0f}  RMSE={rmse_lr:,.0f}  R²={r2_lr:.4f}")

# Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest:      MAE={mae_rf:,.0f}  RMSE={rmse_rf:,.0f}  R²={r2_rf:.4f}")

# Model 3: XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost:            MAE={mae_xgb:,.0f}  RMSE={rmse_xgb:,.0f}  R²={r2_xgb:.4f}")

# Model 4: SARIMA (on aggregate yearly total)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prepare time series: quarterly total arrivals (skip 2021 gap)
agg_ts = df_complete[df_complete['year'] <= TRAIN_END].groupby(
    ['year', 'quarter_num'])['arrivals'].sum().reset_index()
agg_ts = agg_ts.sort_values(['year', 'quarter_num'])
agg_ts = agg_ts[agg_ts['year'] != 2021]  # Remove gap year

ts_data = agg_ts['arrivals'].values

try:
    sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False, maxiter=500)
    
    # Forecast
    n_forecast = len(agg_test)
    sarima_pred = sarima_fit.forecast(steps=n_forecast)
    
    # Align test data (skip 2021 in test too)
    test_ts = df_complete[df_complete['year'] >= TEST_START].groupby(
        ['year', 'quarter_num'])['arrivals'].sum().reset_index()
    test_ts = test_ts.sort_values(['year', 'quarter_num'])
    y_test_sarima = test_ts['arrivals'].values[:len(sarima_pred)]
    
    mae_sarima = mean_absolute_error(y_test_sarima, sarima_pred)
    rmse_sarima = np.sqrt(mean_squared_error(y_test_sarima, sarima_pred))
    r2_sarima = r2_score(y_test_sarima, sarima_pred)
    print(f"SARIMA(1,1,1)(1,1,1,4): MAE={mae_sarima:,.0f}  RMSE={rmse_sarima:,.0f}  R²={r2_sarima:.4f}")
except Exception as e:
    print(f"SARIMA fitting failed: {e}")
    mae_sarima = rmse_sarima = r2_sarima = np.nan
    sarima_pred = None

# Comparison table
print("\n" + "-" * 60)
print("MODEL COMPARISON TABLE")
print("-" * 60)
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'SARIMA'],
    'MAE': [mae_lr, mae_rf, mae_xgb, mae_sarima],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb, rmse_sarima],
    'R²': [r2_lr, r2_rf, r2_xgb, r2_sarima]
})
print(comparison.to_string(index=False))

# ============================================================
# SECTION 6: HYPERPARAMETER OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: HYPERPARAMETER OPTIMIZATION")
print("=" * 60)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
rf_gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_params,
                     cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_gs.fit(X_train, y_train)
y_pred_rf_gs = rf_gs.predict(X_test)
mae_rf_gs = mean_absolute_error(y_test, y_pred_rf_gs)
rmse_rf_gs = np.sqrt(mean_squared_error(y_test, y_pred_rf_gs))
r2_rf_gs = r2_score(y_test, y_pred_rf_gs)
print(f"Best RF params: {rf_gs.best_params_}")
print(f"Optimized RF:   MAE={mae_rf_gs:,.0f}  RMSE={rmse_rf_gs:,.0f}  R²={r2_rf_gs:.4f}")

# RandomizedSearchCV for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
xgb_rs = RandomizedSearchCV(xgb.XGBRegressor(random_state=42), xgb_params,
                            n_iter=50, cv=3, scoring='neg_mean_absolute_error',
                            n_jobs=-1, random_state=42)
xgb_rs.fit(X_train, y_train)
y_pred_xgb_rs = xgb_rs.predict(X_test)
mae_xgb_rs = mean_absolute_error(y_test, y_pred_xgb_rs)
rmse_xgb_rs = np.sqrt(mean_squared_error(y_test, y_pred_xgb_rs))
r2_xgb_rs = r2_score(y_test, y_pred_xgb_rs)
print(f"Best XGB params: {xgb_rs.best_params_}")
print(f"Optimized XGB:   MAE={mae_xgb_rs:,.0f}  RMSE={rmse_xgb_rs:,.0f}  R²={r2_xgb_rs:.4f}")

# Updated comparison table
print("\n" + "-" * 60)
print("UPDATED MODEL COMPARISON TABLE")
print("-" * 60)
comparison_v2 = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'SARIMA',
              'RF (optimized)', 'XGBoost (optimized)'],
    'MAE': [mae_lr, mae_rf, mae_xgb, mae_sarima, mae_rf_gs, mae_xgb_rs],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb, rmse_sarima, rmse_rf_gs, rmse_xgb_rs],
    'R²': [r2_lr, r2_rf, r2_xgb, r2_sarima, r2_rf_gs, r2_xgb_rs]
})
print(comparison_v2.to_string(index=False))

# Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = comparison_v2['Model']
x = range(len(models))

axes[0].bar(x, comparison_v2['MAE'] / 1e6, color=sns.color_palette('Set2', len(models)))
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
axes[0].set_title('MAE (millions)', fontweight='bold')
axes[0].set_ylabel('MAE')

axes[1].bar(x, comparison_v2['RMSE'] / 1e6, color=sns.color_palette('Set2', len(models)))
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
axes[1].set_title('RMSE (millions)', fontweight='bold')
axes[1].set_ylabel('RMSE')

axes[2].bar(x, comparison_v2['R²'], color=sns.color_palette('Set2', len(models)))
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
axes[2].set_title('R² Score', fontweight='bold')
axes[2].set_ylabel('R²')
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ model_comparison.png")

# ============================================================
# SECTION 7: FORECASTING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: FORECASTING NEXT 4 QUARTERS")
print("=" * 60)

# Use the best model (SARIMA) to forecast 2026 Q3-Q4 and 2027 Q1-Q2
# Re-train on full data up to 2026 Q2
full_ts = df_complete.groupby(['year', 'quarter_num'])['arrivals'].sum().reset_index()
full_ts = full_ts.sort_values(['year', 'quarter_num'])
full_ts = full_ts[full_ts['year'] != 2021]
ts_full = full_ts['arrivals'].values

try:
    sarima_full = SARIMAX(ts_full, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
                          enforce_stationarity=False, enforce_invertibility=False)
    sarima_full_fit = sarima_full.fit(disp=False, maxiter=500)
    
    forecast_result = sarima_full_fit.get_forecast(steps=4)
    forecast_mean = np.array(forecast_result.predicted_mean).flatten()
    forecast_ci = np.array(forecast_result.conf_int(alpha=0.05))
    # Map forecast to quarters
    last_year = full_ts['year'].max()
    last_q = full_ts[full_ts['year'] == last_year]['quarter_num'].max()
    
    forecast_quarters = []
    y, q = last_year, last_q
    for _ in range(4):
        q += 1
        if q > 4:
            q = 1
            y += 1
        if y == 2021:  # skip 2021
            q = 1
            y = 2022
        forecast_quarters.append((y, q))
    
    print(f"\nSARIMA Forecast (next 4 quarters):")
    for (y, q), val, ci_lo, ci_hi in zip(forecast_quarters, forecast_mean,
                                          forecast_ci[:, 0], forecast_ci[:, 1]):
        print(f"  {y} Q{q}: {val:,.0f}  [{ci_lo:,.0f} — {ci_hi:,.0f}]")
    
    # Forecast plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical quarterly data
    hist = full_ts.copy()
    hist['label'] = hist['year'].astype(str) + 'Q' + hist['quarter_num'].astype(str)
    ax.plot(range(len(hist)), hist['arrivals'] / 1e6, 'o-', linewidth=2,
            markersize=5, color='#2196F3', label='Historical')
    
    # Forecast
    fc_labels = [f"{y}Q{q}" for y, q in forecast_quarters]
    fc_x = range(len(hist), len(hist) + 4)
    ax.plot(fc_x, forecast_mean / 1e6, 's-', linewidth=2, markersize=8,
            color='#FF5722', label='Forecast')
    ax.fill_between(fc_x, forecast_ci[:, 0] / 1e6, forecast_ci[:, 1] / 1e6,
                    alpha=0.3, color='#FF5722', label='95% CI')
    
    # Labels
    all_labels = list(hist['label']) + fc_labels
    tick_step = max(1, len(all_labels) // 20)
    ax.set_xticks(range(0, len(all_labels), tick_step))
    ax.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), tick_step)],
                       rotation=45, ha='right', fontsize=8)
    
    ax.axvline(x=len(hist) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Total Arrivals (millions)', fontsize=12)
    ax.set_title('Vietnam Tourism Arrivals: Historical & SARIMA Forecast', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/forecast_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ forecast_plot.png")
    
except Exception as e:
    print(f"Forecasting failed: {e}")
    forecast_quarters = []
    forecast_mean = None

# ============================================================
# SECTION 8: SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: SAVING OUTPUTS")
print("=" * 60)

# Save CSVs
df_long.to_csv('output/df_long.csv', index=False)
df_total.to_csv('output/df_total.csv', index=False)
print("✓ df_long.csv")
print("✓ df_total.csv")

# Save comparison table
comparison_v2.to_csv('output/model_comparison.csv', index=False)
print("✓ model_comparison.csv")

# Save forecast
if forecast_mean is not None:
    fc_df = pd.DataFrame({
        'year': [y for y, q in forecast_quarters],
        'quarter': [f'Q{q}' for y, q in forecast_quarters],
        'forecast': forecast_mean,
        'ci_lower': forecast_ci[:, 0],
        'ci_upper': forecast_ci[:, 1],
    })
    fc_df.to_csv('output/forecast.csv', index=False)
    print("✓ forecast.csv")

print("\n✓ All outputs saved successfully!")
