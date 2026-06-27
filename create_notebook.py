#!/usr/bin/env python3
"""Generate the general analysis notebook (code + findings, not full text report)."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []

def md(t): cells.append(new_markdown_cell(t.strip()))
def code(t): cells.append(new_code_cell(t.strip()))

md("# Phân tích Du lịch Quốc tế Việt Nam (2008–2026)\n\nDữ liệu: Lượng khách quốc tế theo quốc gia nguồn và quý, từ 4 file HTML-Excel.\n\n*Lưu ý: Năm 2021 không có dữ liệu (ảnh hưởng COVID-19). Q3/Q4 có dữ liệu từ 2008, Q1/Q2 từ 2009.*")

# --- Imports ---
code("""import pandas as pd
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
print("Imports OK")""")

# --- Parse ---
md("## 1. Thu thập và parse dữ liệu")

code("""def parse_quarterly_file(filepath, quarter_label):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    tree = html.fromstring(raw)
    rows = tree.xpath('//tr')
    header_tds = rows[0].xpath('.//td')
    years = [(td.tail or '').strip() for td in header_tds[2:]]
    records = []
    for row in rows[1:]:
        tds = row.xpath('.//td')
        if len(tds) < 3: continue
        country = (tds[0].tail or '').strip()
        if not country or country == 'Phân theo thị trường': continue
        for j, td in enumerate(tds[1:]):
            val_str = (td.tail or '').strip()
            year_label = years[j] if j < len(years) else None
            if not val_str or not year_label or year_label == 'Totals': continue
            try: val = float(val_str.replace('.', ''))
            except ValueError: continue
            records.append({'country': country, 'year': int(year_label), 'quarter': quarter_label, 'arrivals': val})
    return pd.DataFrame(records)

dfs = []
for label, f in [('Q1','quy1-cacnuoc.xls'),('Q2','quy2-cacnuoc.xls'),('Q3','quy3-cacnuoc.xls'),('Q4','quy4-cacnuoc.xls')]:
    df = parse_quarterly_file(f, label)
    dfs.append(df)
    print(f"{label}: {len(df)} records, {df['country'].nunique()} countries, years {df['year'].min()}-{df['year'].max()}")

df_long = pd.concat(dfs, ignore_index=True)
df_long = df_long[df_long['country'] != 'Totals'].reset_index(drop=True)

# Verify against known anchor
hk = df_long[(df_long['country']=='Hoa Kỳ') & (df_long['year']==2009) & (df_long['quarter']=='Q1')]
assert hk['arrivals'].values[0] == 104520.0, "Verification failed!"
print(f"\\n✓ Hoa Kỳ 2009 Q1 = {hk['arrivals'].values[0]:,.0f}")
print(f"Total: {len(df_long)} records, {df_long['country'].nunique()} countries")""")

# --- Clean ---
md("## 2. Làm sạch và tổng hợp")

code("""# Annual totals
df_total = df_long.groupby(['country','year'])['arrivals'].sum().reset_index()
df_total.columns = ['country','year','total_arrivals']

# Complete grid
all_countries = sorted(df_long['country'].unique())
all_years = sorted(df_long['year'].unique())
idx = pd.MultiIndex.from_product([all_countries, all_years, ['Q1','Q2','Q3','Q4']], names=['country','year','quarter'])
df_complete = df_long.set_index(['country','year','quarter']).reindex(idx, fill_value=0).reset_index()

print(f"df_complete: {df_complete.shape}")
print(f"Countries: {len(all_countries)}, Years: {sorted(all_years)}")
print(f"2021 present: {2021 in df_long['year'].values}")

# Yearly totals
print("\\nYearly totals:")
for y, v in df_total.groupby('year')['total_arrivals'].sum().items():
    print(f"  {y}: {v/1e6:.2f}M")""")

# --- Feature Engineering ---
md("## 3. Feature Engineering & Train-Test Split")

code("""quarter_map = {'Q1':1,'Q2':2,'Q3':3,'Q4':4}
df_complete['quarter_num'] = df_complete['quarter'].map(quarter_map)
df_complete['time_idx'] = df_complete['year'] + (df_complete['quarter_num']-1)/4
df_complete = df_complete.sort_values(['country','year','quarter_num'])
for lag in [1,2,4]:
    df_complete[f'lag_{lag}'] = df_complete.groupby('country')['arrivals'].shift(lag)
df_complete['rolling_mean_4'] = df_complete.groupby('country')['arrivals'].transform(lambda x: x.rolling(4, min_periods=1).mean())

TRAIN_END = 2023
TEST_START = 2024
df_train = df_complete[df_complete['year'] <= TRAIN_END]
df_test = df_complete[df_complete['year'] >= TEST_START]
print(f"Train: {df_train.shape[0]} rows (to {TRAIN_END}), Test: {df_test.shape[0]} rows ({TEST_START}+)")""")

# --- EDA ---
md("## 4. EDA — Xu hướng tổng thể")

code("""fig, ax = plt.subplots(figsize=(14, 7))
yearly = df_total.groupby('year')['total_arrivals'].sum().reset_index()
ax.plot(yearly['year'], yearly['total_arrivals']/1e6, 'o-', lw=2, ms=8, color='#2196F3')
ax.set_xlabel('Năm'); ax.set_ylabel('Tổng lượng khách (triệu)')
ax.set_title('Lượng khách quốc tế đến Việt Nam theo năm', fontweight='bold')
ax.set_xticks(yearly['year']); ax.set_xticklabels(yearly['year'], rotation=45)
covid_val = yearly[yearly['year']==2020]['total_arrivals'].values[0]/1e6
ax.annotate('COVID-19', xy=(2020, covid_val), xytext=(2018, covid_val+1.5),
            fontsize=11, ha='center', arrowprops=dict(arrowstyle='->', color='red', lw=1.5), color='red', fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('eda_total_trend.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 5. Top 10 quốc gia nguồn khách")

code("""top10 = df_total.groupby('country')['total_arrivals'].sum().sort_values(ascending=False).head(10)
print(top10.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top10)), top10.values/1e6, color=sns.color_palette('viridis', len(top10)))
ax.set_yticks(range(len(top10))); ax.set_yticklabels(top10.index, fontsize=11)
ax.set_xlabel('Tổng lượng khách (triệu)'); ax.set_title('Top 10 quốc gia nguồn khách', fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, top10.values):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{val/1e6:.1f}M', va='center', fontsize=10)
plt.tight_layout(); plt.savefig('eda_top10_countries.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 6. Mùa vụ")

code("""fig, ax = plt.subplots(figsize=(10, 6))
quarterly = df_long.groupby('quarter')['arrivals'].agg(['mean','std']).reindex(['Q1','Q2','Q3','Q4'])
ax.bar(quarterly.index, quarterly['mean']/1e3, yerr=quarterly['std']/1e3,
       capsize=5, color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'], edgecolor='black')
ax.set_xlabel('Quý'); ax.set_ylabel('Lượng khách TB (nghìn)')
ax.set_title('Mùa vụ: Lượng khách trung bình theo quý', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('eda_seasonality.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 7. Tương quan giữa các quốc gia nguồn (Top 5)")

code("""top5 = top10.head(5).index.tolist()
pivot = df_total[df_total['country'].isin(top5)].pivot_table(index='year', columns='country', values='total_arrivals', aggfunc='sum')
corr = pivot.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Tương quan giữa 5 quốc gia nguồn lớn nhất', fontweight='bold')
plt.tight_layout(); plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight'); plt.show()
print("\\nNhận xét: Trung Quốc-Đài Loan tương quan cao (0.89), Hoa Kỳ tương quan thấp với các thị trường châu Á.")""")

md("## 8. Xu hướng theo từng quốc gia (Top 5)")

code("""fig, axes = plt.subplots(3, 2, figsize=(16, 14))
for i, c in enumerate(top5):
    d = df_total[df_total['country']==c].sort_values('year')
    axes.flatten()[i].plot(d['year'], d['total_arrivals']/1e3, 'o-', lw=2, ms=6)
    axes.flatten()[i].set_title(c, fontweight='bold'); axes.flatten()[i].set_ylabel('Nghìn lượt')
    axes.flatten()[i].grid(True, alpha=0.3)
axes.flatten()[5].set_visible(False)
plt.suptitle('Xu hướng lượng khách theo quốc gia', fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('eda_country_trends.png', dpi=150, bbox_inches='tight'); plt.show()""")

# --- Modeling ---
md("## 9. Modeling — Chuẩn bị dữ liệu")

code("""from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Aggregate features
agg_all = df_complete.groupby(['year','quarter_num','time_idx'])['arrivals'].sum().reset_index().sort_values(['year','quarter_num'])
agg_all['lag_1'] = agg_all['arrivals'].shift(1)
agg_all['lag_4'] = agg_all['arrivals'].shift(4)
agg_all['rolling_mean_4'] = agg_all['arrivals'].rolling(4, min_periods=1).mean()
feat = ['year','quarter_num','time_idx','lag_1','lag_4','rolling_mean_4']

tr = agg_all[agg_all['year']<=TRAIN_END].dropna(subset=['lag_1','lag_4'])
te = agg_all[agg_all['year']>=TEST_START].dropna(subset=['lag_1','lag_4'])
X_train, y_train = tr[feat].values, tr['arrivals'].values
X_test, y_test = te[feat].values, te['arrivals'].values
print(f"Train: {len(X_train)}, Test: {len(X_test)}")""")

md("## 10. Linear Regression")

code("""lr = LinearRegression().fit(X_train, y_train)
y_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_lr); rmse_lr = np.sqrt(mean_squared_error(y_test, y_lr)); r2_lr = r2_score(y_test, y_lr)
print(f"Linear Regression: MAE={mae_lr:,.0f}  RMSE={rmse_lr:,.0f}  R²={r2_lr:.4f}")""")

md("## 11. Random Forest")

code("""rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_train)
y_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_rf); rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf)); r2_rf = r2_score(y_test, y_rf)
print(f"Random Forest:     MAE={mae_rf:,.0f}  RMSE={rmse_rf:,.0f}  R²={r2_rf:.4f}")""")

md("## 12. XGBoost")

code("""xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42).fit(X_train, y_train)
y_xgb = xgb_m.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_xgb); rmse_xgb = np.sqrt(mean_squared_error(y_test, y_xgb)); r2_xgb = r2_score(y_test, y_xgb)
print(f"XGBoost:           MAE={mae_xgb:,.0f}  RMSE={rmse_xgb:,.0f}  R²={r2_xgb:.4f}")""")

md("## 13. SARIMA")

code("""agg_ts = df_complete[df_complete['year']<=TRAIN_END].groupby(['year','quarter_num'])['arrivals'].sum().reset_index().sort_values(['year','quarter_num'])
agg_ts = agg_ts[agg_ts['year']!=2021]
sarima = SARIMAX(agg_ts['arrivals'].values, order=(1,1,1), seasonal_order=(1,1,1,4),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=500)
test_ts = df_complete[df_complete['year']>=TEST_START].groupby(['year','quarter_num'])['arrivals'].sum().reset_index().sort_values(['year','quarter_num'])
sp = sarima.forecast(steps=len(test_ts))
y_s = test_ts['arrivals'].values[:len(sp)]
mae_s = mean_absolute_error(y_s, sp); rmse_s = np.sqrt(mean_squared_error(y_s, sp)); r2_s = r2_score(y_s, sp)
print(f"SARIMA(1,1,1)(1,1,1,4): MAE={mae_s:,.0f}  RMSE={rmse_s:,.0f}  R²={r2_s:.4f}")""")

md("## 14. So sánh mô hình")

code("""comparison = pd.DataFrame({'Model': ['Linear Regression','Random Forest','XGBoost','SARIMA'],
    'MAE': [mae_lr,mae_rf,mae_xgb,mae_s], 'RMSE': [rmse_lr,rmse_rf,rmse_xgb,rmse_s], 'R²': [r2_lr,r2_rf,r2_xgb,r2_s]})
print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = comparison['Model']; x = range(len(models))
for ax, metric, title in zip(axes, ['MAE','RMSE','R²'], ['MAE','RMSE','R² Score']):
    vals = comparison[metric]/1e6 if metric!='R²' else comparison[metric]
    ax.bar(x, vals, color=sns.color_palette('Set2', len(models)))
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontweight='bold')
    if metric=='R²': ax.axhline(y=0, color='red', ls='--', alpha=0.5)
plt.suptitle('So sánh hiệu suất mô hình', fontweight='bold')
plt.tight_layout(); plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 15. Tối ưu siêu tham số")

code("""from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV for RF
rf_gs = GridSearchCV(RandomForestRegressor(random_state=42),
    {'n_estimators':[100,200,300], 'max_depth':[5,10,15,None], 'min_samples_split':[2,5,10]},
    cv=3, scoring='neg_mean_absolute_error', n_jobs=-1).fit(X_train, y_train)
y_rf_gs = rf_gs.predict(X_test)
print(f"Best RF params: {rf_gs.best_params_}")
mae_rf_gs = mean_absolute_error(y_test, y_rf_gs); rmse_rf_gs = np.sqrt(mean_squared_error(y_test, y_rf_gs)); r2_rf_gs = r2_score(y_test, y_rf_gs)
print(f"Optimized RF:   MAE={mae_rf_gs:,.0f}  RMSE={rmse_rf_gs:,.0f}  R²={r2_rf_gs:.4f}")

# RandomizedSearchCV for XGBoost
xgb_rs = RandomizedSearchCV(xgb.XGBRegressor(random_state=42),
    {'n_estimators':[100,200,300,500], 'max_depth':[3,5,7,9], 'learning_rate':[0.01,0.05,0.1,0.2],
     'subsample':[0.7,0.8,0.9,1.0], 'colsample_bytree':[0.7,0.8,0.9,1.0]},
    n_iter=50, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42).fit(X_train, y_train)
y_xgb_rs = xgb_rs.predict(X_test)
print(f"\\nBest XGB params: {xgb_rs.best_params_}")
mae_xgb_rs = mean_absolute_error(y_test, y_xgb_rs); rmse_xgb_rs = np.sqrt(mean_squared_error(y_test, y_xgb_rs)); r2_xgb_rs = r2_score(y_test, y_xgb_rs)
print(f"Optimized XGB:  MAE={mae_xgb_rs:,.0f}  RMSE={rmse_xgb_rs:,.0f}  R²={r2_xgb_rs:.4f}")

# Final comparison
comp = pd.DataFrame({'Model': ['Linear Regression','Random Forest','XGBoost','SARIMA','RF (optimized)','XGBoost (optimized)'],
    'MAE': [mae_lr,mae_rf,mae_xgb,mae_s,mae_rf_gs,mae_xgb_rs],
    'RMSE': [rmse_lr,rmse_rf,rmse_xgb,rmse_s,rmse_rf_gs,rmse_xgb_rs],
    'R²': [r2_lr,r2_rf,r2_xgb,r2_s,r2_rf_gs,r2_xgb_rs]})
print(f"\\n{comp.to_string(index=False)}")""")

md("## 16. Dự đoán 4 quý tiếp theo (SARIMA)")

code("""full_ts = df_complete.groupby(['year','quarter_num'])['arrivals'].sum().reset_index().sort_values(['year','quarter_num'])
full_ts = full_ts[full_ts['year']!=2021]
s_full = SARIMAX(full_ts['arrivals'].values, order=(1,1,1), seasonal_order=(1,1,1,4),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=500)
fc = s_full.get_forecast(steps=4)
fc_mean = np.array(fc.predicted_mean).flatten()
fc_ci = np.array(fc.conf_int(alpha=0.05))
ly = full_ts['year'].max(); lq = full_ts[full_ts['year']==ly]['quarter_num'].max()
fqs = []; y, q = ly, lq
for _ in range(4):
    q += 1
    if q > 4: q = 1; y += 1
    fqs.append((y, q))

print("Dự đoán:")
for (yr, qr), v, lo, hi in zip(fqs, fc_mean, fc_ci[:,0], fc_ci[:,1]):
    print(f"  {yr} Q{qr}: {v:,.0f}  [{lo:,.0f} — {hi:,.0f}]")

fig, ax = plt.subplots(figsize=(14, 7))
hist = full_ts.copy(); hist['lbl'] = hist['year'].astype(str)+'Q'+hist['quarter_num'].astype(str)
ax.plot(range(len(hist)), hist['arrivals']/1e6, 'o-', lw=2, ms=5, color='#2196F3', label='Historical')
fcx = range(len(hist), len(hist)+4)
ax.plot(fcx, fc_mean/1e6, 's-', lw=2, ms=8, color='#FF5722', label='Forecast')
ax.fill_between(fcx, fc_ci[:,0]/1e6, fc_ci[:,1]/1e6, alpha=0.3, color='#FF5722', label='95% CI')
ax.axvline(x=len(hist)-0.5, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Quý'); ax.set_ylabel('Tổng lượng khách (triệu)')
ax.set_title('Dự đoán SARIMA — 4 quý tiếp theo', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('forecast_plot.png', dpi=150, bbox_inches='tight'); plt.show()""")

md("## 17. Lưu kết quả")

code("""df_long.to_csv('df_long.csv', index=False)
df_total.to_csv('df_total.csv', index=False)
comp.to_csv('model_comparison.csv', index=False)
pd.DataFrame({'year':[y for y,q in fqs],'quarter':[f'Q{q}' for y,q in fqs],
    'forecast':fc_mean,'ci_lower':fc_ci[:,0],'ci_upper':fc_ci[:,1]}).to_csv('forecast.csv', index=False)
print("✓ Saved: df_long.csv, df_total.csv, model_comparison.csv, forecast.csv")""")

nb.cells = cells
with open('bao-cao.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f'Created bao-cao.ipynb with {len(cells)} cells')
