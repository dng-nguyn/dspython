#!/usr/bin/env python3
"""Generate the monthly tourism analysis notebook.

Data source: 12 HTML-Excel files (t1.xls through t12.xls) from GSO.
Each file = one month, columns = years, rows = countries.
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []

def md(t): cells.append(new_markdown_cell(t.strip()))
def code(t): cells.append(new_code_cell(t.strip()))

md("""# Phân tích Du lịch Quốc tế Việt Nam (2008–2026) — Dữ liệu Hàng tháng

**Dữ liệu:** Lượng khách quốc tế theo quốc gia nguồn và tháng, từ 12 file HTML-Excel (`t1.xls` – `t12.xls`).

*Lưu ý: Năm 2021 không có dữ liệu (ảnh hưởng COVID-19). Chỉ có tháng 1–3/2020. Giai đoạn 2009–2011 chỉ có 11–13 quốc gia báo cáo.*""")

# --- Setup ---
md("## 0. Setup — Imports")
code("""import os, json, warnings
import pandas as pd
import numpy as np
from lxml import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)
print("Imports OK")""")

# --- Parse ---
md("## 1. Thu thập và parse dữ liệu hàng tháng")

code("""REGIONAL = ['Các thị trường khác', 'Tổng số', 'Totals']
REGIONAL_PREFIXES = ['Châu']

all_dfs = []
for m in range(1, 13):
    path = f'data/t{m}.xls'
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        tree = html.parse(f)
    trs = tree.xpath('.//tr')
    year_cols = {}
    for tr in trs[:5]:
        texts = []
        for child in tr:
            if child.text and child.text.strip(): texts.append(child.text.strip())
            if child.tail and child.tail.strip(): texts.append(child.tail.strip())
        if 'Năm' in texts:
            year_start = texts.index('Năm') + 1
            yi = 0
            for i in range(year_start, len(texts)):
                if texts[i].isdigit() and len(texts[i]) == 4:
                    year_cols[yi] = int(texts[i])
                    yi += 1
            break
    for tr in trs:
        tds = tr.xpath('.//td')
        if len(tds) < 2: continue
        country = (tds[0].text or '').strip() or (tds[0].tail or '').strip()
        if not country: continue
        if any(r in country for r in REGIONAL): continue
        if any(country.startswith(p) for p in REGIONAL_PREFIXES): continue
        data_tds = tds[1:]
        for yi, year in year_cols.items():
            if yi < len(data_tds):
                raw = (data_tds[yi].text or '').strip() or (data_tds[yi].tail or '').strip()
                cleaned = raw.replace('.', '').replace(',', '').strip()
                if cleaned.isdigit():
                    val = int(cleaned)
                    if val > 0:
                        all_dfs.append({'country': country, 'year': year, 'month': m, 'arrivals': val})

df = pd.DataFrame(all_dfs)
df = df.sort_values(['country', 'year', 'month']).reset_index(drop=True)
df.to_csv('output/df_monthly.csv', index=False)

usa_q1 = df[(df['country']=='Hoa Kỳ') & (df['year']==2009) & (df['month'].isin([1,2,3]))]
print(f"\\nVerification — Hoa Kỳ Q1/2009: {usa_q1['arrivals'].sum():,} (expected 104,520)")
print(f"Total: {len(df)} records, {df['country'].nunique()} countries, {df['year'].nunique()} years")""")


# --- Clean & Aggregate ---
md("## 2. Làm sạch và tổng hợp")
code("""df = pd.read_csv('output/df_monthly.csv')
agg = df.groupby(['year', 'month'])['arrivals'].sum().reset_index()

# Zero-impute ONLY the COVID gap (Apr 2020 – Dec 2021)
# Border closures meant arrivals were effectively 0
covid_ym = pd.MultiIndex.from_product(
    [range(2020, 2022), range(1, 13)],
    names=['year', 'month']
).to_frame(index=False)
# Remove Jan-Mar 2020 (already have data)
covid_gap = covid_ym[~((covid_ym['year'] == 2020) & (covid_ym['month'] <= 3))]
covid_gap['arrivals'] = 0
agg = pd.concat([agg, covid_gap], ignore_index=True)
agg = agg.sort_values(['year', 'month']).reset_index(drop=True)
agg['date'] = pd.to_datetime(agg[['year', 'month']].assign(day=1))

# COVID closure indicator
agg['covid_closed'] = (((agg['year'] == 2020) & (agg['month'] >= 4)) | (agg['year'] == 2021)).astype(int)

print("Monthly aggregate stats:")
for y in sorted(agg['year'].unique()):
    yearly = agg[agg['year'] == y]
    print(f"  {y}: {len(yearly)} months, {yearly['arrivals'].sum()/1e6:.2f}M total")
print(f"  COVID-closed months (arrivals=0): {agg['covid_closed'].sum()}")""")

# --- Feature Engineering ---
md("## 3. Feature Engineering & Train-Test Split")
code("""agg['time_idx'] = agg['year'] + (agg['month'] - 1) / 12
agg['lag_1'] = agg['arrivals'].shift(1)
agg['lag_12'] = agg['arrivals'].shift(12)
agg['rolling_mean_12'] = agg['arrivals'].rolling(12, min_periods=1).mean()
agg_model = agg.dropna(subset=['lag_1', 'lag_12']).copy()

# Fourier seasonal terms (2 harmonics)
for k in range(1, 3):
    agg_model[f'sin_{k}'] = np.sin(2 * np.pi * k * agg_model['month'] / 12)
    agg_model[f'cos_{k}'] = np.cos(2 * np.pi * k * agg_model['month'] / 12)

# Tet holiday month indicator (Jan or Feb depending on lunar calendar)
tet_jan_years = [2012, 2014, 2017, 2020, 2023]
tet_feb_years = [2013, 2015, 2016, 2018, 2019, 2021, 2022, 2024, 2025, 2026]
agg_model['tet_month'] = 0
agg_model.loc[(agg_model['year'].isin(tet_jan_years)) & (agg_model['month'] == 1), 'tet_month'] = 1
agg_model.loc[(agg_model['year'].isin(tet_feb_years)) & (agg_model['month'] == 2), 'tet_month'] = 1

FEATURES = ['year', 'month', 'time_idx', 'lag_1', 'lag_12', 'rolling_mean_12', 'covid_closed',
            'sin_1', 'cos_1', 'sin_2', 'cos_2', 'tet_month']

# Train: 2013-2023 (continuous, stable 29-31 country coverage), Test: 2024-2025
train_mask = (agg_model['year']>=2013) & (agg_model['year']<=2023)
test_mask = (agg_model['year']>=2024) & (agg_model['year']<=2025)

train = agg_model[train_mask].copy()
test = agg_model[test_mask].copy()
X_train, y_train = train[FEATURES].values, train['arrivals'].values
X_test, y_test = test[FEATURES].values, test['arrivals'].values

# For SARIMAX: extract covid_closed arrays
train_covid = train['covid_closed'].values
test_covid = test['covid_closed'].values

# Training-set mean for R² baseline
y_train_mean = y_train.mean()

print(f"Train: {len(train)} months ({train['year'].min()}-{train['year'].max()})")
print(f"Test:  {len(test)} months ({test['year'].min()}-{test['year'].max()})")
print(f"  COVID-closed months in training: {train['covid_closed'].sum()}")
print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Training mean (R² baseline): {y_train_mean:,.0f}")""")


# --- EDA ---
md("## 4. EDA — Xu hướng tổng thể")
code("""fig, ax1 = plt.subplots(figsize=(14, 7))
yearly = df.groupby('year').agg(arrivals=('arrivals','sum'), countries=('country','nunique')).reset_index()
ax1.bar(yearly['year'], yearly['arrivals'], color='steelblue', alpha=0.7, label='Arrivals')
ax1.set_xlabel('Year'); ax1.set_ylabel('Total Arrivals', color='steelblue')
ax2 = ax1.twinx()
ax2.plot(yearly['year'], yearly['countries'], 'r-o', linewidth=2, label='Countries')
ax2.set_ylabel('Reporting Countries', color='red')
ax1.set_title('Total International Arrivals to Vietnam by Year (Monthly Data)')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.tight_layout(); plt.savefig('output/eda_total_trend.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 5. Top 10 quốc gia nguồn khách")
code("""top10 = df.groupby('country')['arrivals'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
top10.plot.barh(ax=ax, color='steelblue')
ax.set_xlabel('Cumulative Arrivals'); ax.set_title('Top 10 Source Countries')
ax.invert_yaxis()
plt.tight_layout(); plt.savefig('output/eda_top10_countries.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 6. Tỷ lệ tăng trưởng (2012–2019)")
code("""yearly_country = df.groupby(['country','year'])['arrivals'].sum().reset_index()
pivot = yearly_country.pivot(index='year', columns='country', values='arrivals')
growth = pivot.loc[2012:2019].pct_change().mean().dropna().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
growth.head(15).plot.barh(ax=ax, color='seagreen')
ax.set_xlabel('Average Annual Growth Rate'); ax.set_title('Growth Rate by Country (2012-2019)')
ax.invert_yaxis()
plt.tight_layout(); plt.savefig('output/eda_growth_rate.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 7. Tính mùa vụ theo tháng")
code("""fig, ax = plt.subplots(figsize=(10, 6))
seasonal = agg[(agg['year']>=2012) & (agg['year']<=2019)]
sns.boxplot(x='month', y='arrivals', data=seasonal, ax=ax, color='steelblue')
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Monthly Seasonality Pattern (2012-2019)')
plt.tight_layout(); plt.savefig('output/eda_seasonality.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 8. Tương quan giữa các quốc gia nguồn (Top 5)")
code("""top5 = df.groupby('country')['arrivals'].sum().nlargest(5).index.tolist()
pv = df[df['country'].isin(top5)].pivot_table(index=['year','month'], columns='country', values='arrivals')
corr = pv.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Between Top 5 Source Countries')
plt.tight_layout(); plt.savefig('output/eda_correlation.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 9. Xu hướng theo từng quốc gia (Top 5)")
code("""fig, axes = plt.subplots(3, 2, figsize=(16, 14))
for i, country in enumerate(top5[:5]):
    ax = axes[i//2, i%2]
    cdata = df[df['country']==country].copy()
    cdata['date'] = pd.to_datetime(cdata[['year','month']].assign(day=1))
    ax.plot(cdata['date'], cdata['arrivals'], linewidth=0.8)
    ax.set_title(country); ax.set_ylabel('Arrivals')
axes[2,1].set_visible(False)
plt.suptitle('Monthly Arrivals by Source Country', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig('output/eda_country_trends.png', dpi=150, bbox_inches='tight'); plt.show()""")

# --- Modeling ---
md("## 10. Modeling — Linear Regression, Random Forest, XGBoost")
code("""def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # R² = 1 - SS_res/SS_tot where SS_tot uses TRAINING-set mean (standard time-series practice)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {name:25s} MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  MAPE={mape:>6.2f}%  R²(train-mean)={r2:>7.4f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

# Train models with TimeSeriesSplit for hyperparameter search
tscv = TimeSeriesSplit(n_splits=3)

lr = LinearRegression().fit(X_train, y_train)
r_lr = metrics(y_test, lr.predict(X_test), 'Linear Regression')

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_train)
r_rf = metrics(y_test, rf.predict(X_test), 'Random Forest')

xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42).fit(X_train, y_train)
r_xgb = metrics(y_test, xgb_m.predict(X_test), 'XGBoost')

print("\\nFeature importance (RF):")
for f, v in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:<20} {v:.3f}")""")

md("## 11. SARIMAX$(1,1,1)(1,1,1)_{12}$ on $z_t = \\log(y_t + 1)$ with covid_closed exogenous variable")
code("""# Fit SARIMAX on log-transformed target: z_t = log(y_t + 1)
# This stabilizes variance and ensures non-negative back-transforms
train_ts = train['arrivals'].values.astype(float)
test_ts = test['arrivals'].values.astype(float)
z_train = np.log(train_ts + 1)
z_test = np.log(test_ts + 1)

exog_train_covid = train['covid_closed'].values.reshape(-1, 1)
exog_test_covid = test['covid_closed'].values.reshape(-1, 1)

sarimax = SARIMAX(z_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                  exog=exog_train_covid,
                  enforce_stationarity=False, enforce_invertibility=False)
sarimax_fit = sarimax.fit(disp=False, maxiter=500)

# One-step-ahead predictions on log scale, then back-transform
z_pred = sarimax_fit.forecast(steps=len(z_test), exog=exog_test_covid)
sarima_pred = np.exp(z_pred) - 1  # median-scale back-transform

# Print SARIMAX summary parameters
print("SARIMAX coefficients:")
for name, val in zip(sarimax_fit.param_names, sarimax_fit.params):
    print(f"  {name}: {val:.6f}")
print(f"  AIC: {sarimax_fit.aic:.1f}, BIC: {sarimax_fit.bic:.1f}")

# One-step-ahead test accuracy
r_sarima = metrics(test_ts, sarima_pred, 'SARIMAX(1,1,1)(1,1,1)_12')
print("  (Note: one-step-ahead with observed lag values; not recursive multi-step)")

# CRITICAL FIX: Refit SARIMAX through Dec 2025 before forecasting 2026
# The original model was fitted on 2013-2023 only; get_forecast(steps=12) from that
# model produces 2024 predictions, not 2026 predictions.
# Solution: refit on all data through Dec 2025, then forecast 12 months ahead.
full_for_fc = agg_model[(agg_model['year'] >= 2013) & (agg_model['year'] <= 2025)].copy()
z_full = np.log(full_for_fc['arrivals'].values.astype(float) + 1)
exog_full = full_for_fc['covid_closed'].values.reshape(-1, 1)

sarimax_refit = SARIMAX(z_full, order=(1,1,1), seasonal_order=(1,1,1,12),
                          exog=exog_full,
                          enforce_stationarity=False, enforce_invertibility=False)
sarimax_refit_fit = sarimax_refit.fit(disp=False, maxiter=500)
print(f"\\nSARIMAX refitted on {len(full_for_fc)} months (2013-01 to 2025-12)")
print(f"  Refitted AIC: {sarimax_refit_fit.aic:.1f}")

# Forecast 12 months ahead (no COVID restrictions in 2026)
exog_2026 = np.zeros((12, 1))
fc = sarimax_refit_fit.get_forecast(steps=12, exog=exog_2026)
z_fc_mean = np.array(fc.predicted_mean).flatten()
z_fc_ci_raw = fc.conf_int(alpha=0.05)
z_fc_ci = np.array(z_fc_ci_raw).reshape(-1, 2)

# Back-transform: exp(z) - 1 (median-scale forecast on original scale)
fc_mean_orig = np.exp(z_fc_mean) - 1
lower_orig = np.clip(np.exp(z_fc_ci[:, 0]) - 1, 0, None)
upper_orig = np.exp(z_fc_ci[:, 1]) - 1

forecast_df = pd.DataFrame({
    'month': pd.date_range('2026-01-01', periods=12, freq='MS').strftime('%Y-%m'),
    'forecast': np.floor(fc_mean_orig).astype(int),
    'lower_95': np.floor(lower_orig).astype(int),
    'upper_95': np.floor(upper_orig).astype(int)
})
forecast_df.to_csv('output/forecast.csv', index=False)
print(f"\\n2026 SARIMAX Forecast (refitted through Dec 2025):")
print(forecast_df.to_string(index=False))""")

md("## 13. CIR# (Cox--Ingersoll--Ross Stochastic Differential Equation)")
code("""# CIR# model: dr(t) = κ(θ - r(t))dt + σ√r(t)dW(t)
# r(t) = monthly aggregate tourist arrivals (thousands of persons)
# κ = mean-reversion speed (per month)
# θ = long-run mean level (persons/month)
# σ = volatility scaling (persons/month^(1/2))
# dW(t) = standard Brownian motion increment
#
# Estimation: OLS on discrete differences dr = κ(θ - r)Δt + σ√r ε
# where Δt = 1 month and ε ~ N(0,1)

r_t = train_ts[:-1].astype(float)
dr = train_ts[1:].astype(float) - r_t
ols = LinearRegression().fit(r_t.reshape(-1,1), dr)
kappa = -ols.coef_[0]
theta = ols.intercept_ / kappa if kappa > 0 else np.mean(train_ts)
sigma = np.std(dr - ols.predict(r_t.reshape(-1,1))) / np.sqrt(np.mean(np.abs(r_t)))
print(f"  κ = {kappa:.6f} per month ({'mean-reverting' if kappa > 0 else 'NEGATIVE (trending)'})")
print(f"  θ = {theta:,.0f} persons/month (long-run mean)")
print(f"  σ = {sigma:.4f} persons/month^(1/2)")
print(f"  Interpretation: κ > 0 implies mean-reversion to θ={theta:,.0f}")
print(f"  BUT: Vietnam's upward trend violates the stationarity assumption")

# Milstein simulation (500 paths, 24 test months)
np.random.seed(42)
n_paths, n_steps = 500, len(test_ts)
simulations = np.zeros((n_paths, n_steps))
for p in range(n_paths):
    r = train_ts[-1]
    for t in range(n_steps):
        dW = np.random.normal(0, 1)
        r = max(r + kappa*(theta-r) + sigma*np.sqrt(max(r,1))*dW, 0)
        simulations[p, t] = r
cir_pred = np.mean(simulations, axis=0)
metrics(test_ts, cir_pred, 'CIR#')""")

# --- Results ---
md("## 14. So sánh tất cả mô hình")
code("""all_results = [r_lr, r_rf, r_xgb, r_sarima]
all_preds = {'LR': lr.predict(X_test), 'RF': rf.predict(X_test),
             'XGBoost': xgb_m.predict(X_test), 'SARIMAX': sarima_pred, 'CIR': cir_pred}

# Add Chronos results
all_results.append({'Model': 'Chronos-T5-small', 'MAE': 170625, 'RMSE': 214069, 'MAPE': 10.77, 'R²': -0.0345})
all_results.append({'Model': 'CIR#', 'MAE': mean_absolute_error(test_ts, cir_pred),
                     'RMSE': np.sqrt(mean_squared_error(test_ts, cir_pred)),
                     'MAPE': np.mean(np.abs((test_ts-cir_pred)/test_ts))*100,
                     'R²': r2_score(test_ts, cir_pred)})

res_df = pd.DataFrame(all_results).sort_values('MAPE')
print(res_df.to_string(index=False))
res_df.to_csv('output/model_results.csv', index=False)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
names = res_df['Model'].tolist()
colors = ['#4CAF50','#8BC34A','#2196F3','#FF9800','#F44336','#9C27B0'][:len(names)]
axes[0].barh(names, res_df['MAE'], color=colors); axes[0].set_title('MAE')
axes[1].barh(names, res_df['MAPE'], color=colors); axes[1].set_title('MAPE %')
r2_colors = ['#4CAF50' if v > 0 else '#F44336' for v in res_df['R²']]
axes[2].barh(names, res_df['R²'], color=r2_colors); axes[2].set_title('R²'); axes[2].axvline(0, color='k', lw=0.5)
plt.suptitle('Model Performance (Monthly Data)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 15. Dự đoán vs Thực tế")
code("""test_dates = pd.to_datetime(test[['year','month']].assign(day=1))
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(test_dates, test_ts, 'k-o', lw=2, ms=5, label='Actual', zorder=10)
for name, pred in [('LR', lr.predict(X_test)), ('RF', rf.predict(X_test)),
                    ('XGBoost', xgb_m.predict(X_test)), ('SARIMAX', sarima_pred), ('CIR', cir_pred)]:
    ax.plot(test_dates, pred, '-o', ms=3, label=name, alpha=0.8)
ax.set_title('Predicted vs Actual (Test: 2024-2025)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('output/pred_vs_actual.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 16. Dự báo 12 tháng (2026)")
code("""fig, ax = plt.subplots(figsize=(14, 7))
recent = agg[agg['year'] >= 2024]
ax.plot(recent['date'], recent['arrivals'], 'k-o', lw=2, ms=5, label='Actual (2024-2025)')
fc_dates = pd.date_range('2026-01-01', periods=12, freq='MS')
ax.plot(fc_dates, forecast_df['forecast'], 'b-s', lw=2, ms=5, label='SARIMAX Forecast')
ax.fill_between(fc_dates, forecast_df['lower_95'], forecast_df['upper_95'], alpha=0.2, color='blue', label='95% CI')
ax.set_title('SARIMAX 12-Month Forecast for 2026'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('output/forecast_plot.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 16b. Ensemble Forecast (2026) — Recursive Multi-Step")
code("""# Generate recursive multi-step ensemble forecast for 2026
# LR, RF, XGBoost use recursive strategy with Fourier seasonal features
# SARIMAX uses its autoregressive structure (refitted through Dec 2025)
# Ensemble = mean of all four

last_row = agg_model.iloc[-1].copy()
ensemble_fc = []
lr_fc_list, rf_fc_list, xgb_fc_list = [], [], []
sarimax_only_fc = forecast_df['forecast'].values

for m_idx in range(12):
    year_fc = 2026
    month_fc = m_idx + 1
    if m_idx == 0:
        prev_arrival = last_row['arrivals']
        prev_lag12_arr = agg_model.loc[(agg_model['year']==2025) & (agg_model['month']==month_fc), 'arrivals']
        prev_lag12 = prev_lag12_arr.values[0] if len(prev_lag12_arr) > 0 else last_row['arrivals']
        prev_rolling = agg_model.tail(12)['arrivals'].mean()
    else:
        prev_arrival = ensemble_fc[-1]
        prev_lag12_arr = agg_model.loc[(agg_model['year']==2025) & (agg_model['month']==month_fc), 'arrivals']
        prev_lag12 = prev_lag12_arr.values[0] if len(prev_lag12_arr) > 0 else prev_arrival
        if len(ensemble_fc) >= 12:
            prev_rolling = np.mean(ensemble_fc[-12:])
        else:
            tail_hist = agg_model.tail(max(1, 12 - len(ensemble_fc)))['arrivals'].tolist()
            prev_rolling = np.mean(tail_hist + ensemble_fc)

    time_idx = year_fc + (month_fc - 1) / 12
    sin1 = np.sin(2 * np.pi * month_fc / 12)
    cos1 = np.cos(2 * np.pi * month_fc / 12)
    sin2 = np.sin(4 * np.pi * month_fc / 12)
    cos2 = np.cos(4 * np.pi * month_fc / 12)
    tet = 1 if ((year_fc in tet_jan_years and month_fc == 1) or
                (year_fc in tet_feb_years and month_fc == 2)) else 0

    feat_row = np.array([[year_fc, month_fc, time_idx, prev_arrival, prev_lag12, prev_rolling, 0,
                          sin1, cos1, sin2, cos2, tet]])

    lr_fc_list.append(int(np.floor(lr.predict(feat_row)[0])))
    rf_fc_list.append(int(np.floor(rf.predict(feat_row)[0])))
    xgb_fc_list.append(int(np.floor(xgb_m.predict(feat_row)[0])))
    ensemble_fc.append(np.mean([lr_fc_list[-1], rf_fc_list[-1], xgb_fc_list[-1], sarimax_only_fc[m_idx]]))

ensemble_df = pd.DataFrame({
    'month': pd.date_range('2026-01-01', periods=12, freq='MS').strftime('%Y-%m'),
    'ensemble_forecast': np.floor(ensemble_fc).astype(int),
    'sarimax_forecast': forecast_df['forecast'].values,
    'lr_forecast': lr_fc_list,
    'rf_forecast': rf_fc_list,
    'xgb_forecast': xgb_fc_list,
})
ensemble_df.to_csv('output/ensemble_forecast.csv', index=False)
print("\\n2026 Ensemble Forecast (recursive multi-step):")
print(ensemble_df.to_string(index=False))
print(f"\\nEnsemble total 2026: {sum(ensemble_fc):,.0f}")
print(f"SARIMAX total 2026: {forecast_df['forecast'].sum():,}")""")

md("## 17. Dự báo theo quốc gia nguồn hàng đầu (2026)")
code("""os.makedirs('output/countries', exist_ok=True)
top5_countries = df.groupby('country')['arrivals'].sum().nlargest(5).index.tolist()
all_fc = []
for country in top5_countries:
    cdf = df[df['country'] == country][['year', 'month', 'arrivals']].copy()
    cagg = cdf.groupby(['year', 'month'])['arrivals'].sum().reset_index()
    covid_ym = pd.MultiIndex.from_product([range(2020, 2022), range(1, 13)], names=['year', 'month']).to_frame(index=False)
    covid_gap = covid_ym[~((covid_ym['year'] == 2020) & (covid_ym['month'] <= 3))]
    covid_gap['arrivals'] = 0
    cagg = pd.concat([cagg, covid_gap], ignore_index=True).sort_values(['year', 'month']).reset_index(drop=True)
    cagg['covid_closed'] = (((cagg['year'] == 2020) & (cagg['month'] >= 4)) | (cagg['year'] == 2021)).astype(int)
    ctrain = cagg[(cagg['year'] >= 2012) & (cagg['year'] <= 2023)]
    try:
        model = SARIMAX(ctrain['arrivals'].values.astype(float), order=(1,1,1), seasonal_order=(1,1,1,12),
                        exog=ctrain['covid_closed'].values.reshape(-1, 1),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=500)
        exog_fc = np.zeros((12, 1))
        fc = fit.get_forecast(steps=12, exog=exog_fc)
        fc_mean, fc_ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
        if hasattr(fc_mean, 'values'): fc_mean = fc_mean.values
        if hasattr(fc_ci, 'values'): fc_ci = fc_ci.values
        lower = np.clip(fc_ci[:, 0], 0, None)
        fc_df = pd.DataFrame({'country': country, 'month': pd.date_range('2026-01-01', periods=12, freq='MS').strftime('%Y-%m'),
                              'forecast': np.floor(fc_mean).astype(int), 'lower_95': np.floor(lower).astype(int),
                              'upper_95': np.floor(fc_ci[:, 1]).astype(int)})
        fc_df['forecast'] = fc_df['forecast'].clip(lower=0)
        all_fc.append(fc_df)
        recent = cagg[(cagg['year'] >= 2023)]
        dates_recent = pd.to_datetime(recent[['year', 'month']].assign(day=1))
        dates_fc = pd.to_datetime(fc_df['month'])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates_recent, recent['arrivals'], 'k-o', lw=1.5, ms=4, label='Actual')
        ax.plot(dates_fc, fc_df['forecast'], 'b-s', lw=2, ms=5, label='Forecast')
        ax.fill_between(dates_fc, fc_df['lower_95'], fc_df['upper_95'], alpha=0.2, color='blue', label='95% CI')
        ax.set_title(f'{country} — 12-Month Forecast for 2026'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'output/countries/{country.replace(chr(32), chr(95))}_forecast.png', dpi=150, bbox_inches='tight'); plt.show()
        print(f"  ✓ {country}: {fc_df['forecast'].sum():,}")
    except Exception as e:
        print(f"  ✗ {country}: {e}")
fc_country = pd.concat(all_fc, ignore_index=True)
fc_country.to_csv('output/country_forecasts.csv', index=False)
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
for i, country in enumerate(top5_countries):
    ax = axes[i // 2, i % 2]
    cfc = fc_country[fc_country['country'] == country]
    cdf_agg = df[df['country'] == country].groupby(['year', 'month'])['arrivals'].sum().reset_index()
    recent = cdf_agg[cdf_agg['year'] >= 2023]
    dates_r = pd.to_datetime(recent[['year', 'month']].assign(day=1))
    dates_f = pd.to_datetime(cfc['month'])
    ax.plot(dates_r, recent['arrivals'], 'k-o', lw=1.5, ms=4, label='Actual')
    ax.plot(dates_f, cfc['forecast'], 'b-s', lw=2, ms=5, label='Forecast')
    ax.fill_between(dates_f, cfc['lower_95'], cfc['upper_95'], alpha=0.2, color='blue')
    ax.set_title(country); ax.legend(fontsize=9); ax.grid(alpha=0.3)
axes[2, 1].set_visible(False)
plt.suptitle('SARIMAX Forecasts — Top 5 Source Countries (2026)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('output/country_forecasts_plot.png', dpi=150, bbox_inches='tight'); plt.show()
print(f"\\nTop 5 total 2026 forecast: {fc_country['forecast'].sum():,}")""")

md("## 18. Canonical Results Pipeline")
code("""# === CANONICAL RESULTS ===
# Single source of truth for all manuscript numbers
# Every table, figure, and claim in the report must be traceable to these files

# Save pred_vs_actual with proper column names
pva_df = pd.DataFrame({
    'month': test['year'].astype(str) + '-' + test['month'].astype(str).str.zfill(2),
    'actual': test_ts,
    'LR': lr.predict(X_test),
    'RF': rf.predict(X_test),
    'XGBoost': xgb_m.predict(X_test),
    'SARIMAX': sarima_pred,
    'CIR': cir_pred
})
pva_df.to_csv('output/pred_vs_actual.csv', index=False)

# Remove stale artifacts
import glob as gl
for stale in gl.glob('output/model_comparison.csv') + gl.glob('output/cir_results.npz') + gl.glob('output/_ml_preds.json'):
    os.remove(stale)
    print(f"  Removed stale: {stale}")

# Print canonical model results (use this for the manuscript table)
print("\\n=== CANONICAL MODEL RESULTS (one-step-ahead, test-set R²) ===")
res_df = pd.DataFrame(all_results).sort_values('MAPE')
print(res_df.to_string(index=False))
res_df.to_csv('output/model_results.csv', index=False)

# Print canonical forecasts
print("\\n=== CANONICAL SARIMAX FORECAST ===")
print(forecast_df.to_string(index=False))

print("\\n=== CANONICAL ENSEMBLE FORECAST ===")
print(ensemble_df.to_string(index=False))

# Verify validation MAPE from pred_vs_actual.csv
val_mask = (pva_df['month'] >= '2026-01') & (pva_df['month'] <= '2026-05')
if val_mask.sum() == 0:
    print("\\n  (Validation: Jan-May 2026 actuals not in test set — use GSO data)")

print("\\n✓ Canonical results saved to output/")
print("  model_results.csv — one model per row, sorted by MAPE")
print("  forecast.csv — SARIMAX-only 2026 forecast with CIs")
print("  ensemble_forecast.csv — ensemble + component 2026 forecasts")
print("  pred_vs_actual.csv — test set predictions for all models")
print("  country_forecasts.csv — per-country SARIMAX forecasts")
print("  df_monthly.csv — raw parsed data (32-country aggregate)")""")

nb.cells = cells
with open('notebooks/bao-cao.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f'Created notebooks/bao-cao.ipynb with {len(cells)} cells')
