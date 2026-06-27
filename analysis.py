#!/usr/bin/env python3
"""
Vietnam International Tourism Arrivals Analysis
Data: Quarterly arrivals by country (Q1-Q4), 2008-2026
"""
# %% [markdown]
# # Phân tích Du lịch Quốc tế Việt Nam
# ## Vietnam International Tourism Arrivals Analysis

# %% [code] Cell 1: Imports
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
print("All imports successful.")

# %% [code] Cell 2: Parse quarterly HTML-Excel files
def parse_quarterly_file(filepath, quarter_label):
    """Parse an HTML-Excel .xls file with Vietnamese tourism data.
    
    Structure: header row has years in td[2:].tail, data rows have
    country name in td[0].tail (colspan=2) and values in td[1:].tail.
    Numbers use Vietnamese format: dots as thousands separators.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    tree = html.fromstring(raw)
    rows = tree.xpath('//tr')
    
    # Extract year labels from header (row 0, td[2:])
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
    ('Q1', 'quy1-cacnuoc.xls'),
    ('Q2', 'quy2-cacnuoc.xls'),
    ('Q3', 'quy3-cacnuoc.xls'),
    ('Q4', 'quy4-cacnuoc.xls'),
]

dfs = []
for label, fpath in files:
    df = parse_quarterly_file(fpath, label)
    dfs.append(df)
    print(f"{label}: {len(df)} records, {df['country'].nunique()} countries, "
          f"years {df['year'].min()}-{df['year'].max()}")

df_long = pd.concat(dfs, ignore_index=True)

# Verification: Hoa Kỳ 2009 Q1 should be 104,520
hk_check = df_long[(df_long['country'] == 'Hoa Kỳ') & 
                    (df_long['year'] == 2009) & (df_long['quarter'] == 'Q1')]
assert hk_check['arrivals'].values[0] == 104520.0, "Hoa Kỳ verification failed!"
print(f"\n✓ Verification passed: Hoa Kỳ 2009 Q1 = {hk_check['arrivals'].values[0]:,.0f}")
print(f"\ndf_long shape: {df_long.shape}")
print(f"Countries: {df_long['country'].nunique()}")
print(f"Years: {sorted(df_long['year'].unique())}")

# %% [code] Cell 3: Create df_total (annual arrivals by country)
df_total = df_long.groupby(['country', 'year'])['arrivals'].sum().reset_index()
df_total.columns = ['country', 'year', 'total_arrivals']
df_total = df_total.sort_values(['country', 'year']).reset_index(drop=True)

print(f"df_total shape: {df_total.shape}")
print(f"\nTop 10 countries by total arrivals (all years combined):")
top10 = df_total.groupby('country')['total_arrivals'].sum().sort_values(ascending=False).head(10)
print(top10.to_string())

print(f"\nTotal arrivals by year:")
yearly = df_total.groupby('year')['total_arrivals'].sum()
print(yearly.to_string())

# %% [code] Cell 4: Data quality report
print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

# Missing values per column in df_long
print("\n1. Missing values in df_long:")
print(df_long.isnull().sum())

# Year coverage by quarter
print("\n2. Year coverage by quarter:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    years_q = sorted(df_long[df_long['quarter'] == q]['year'].unique())
    print(f"  {q}: {years_q}")

# 2021 check
has_2021 = 2021 in df_long['year'].values
print(f"\n3. 2021 data present: {has_2021}")

# Countries with sparse data
print("\n4. Countries with fewer than 10 year-quarter records:")
counts = df_long.groupby('country').size().sort_values()
sparse = counts[counts < 10]
if len(sparse) > 0:
    print(sparse.to_string())
else:
    print("  None")

# Duplicates
dups = df_long.duplicated(subset=['country', 'year', 'quarter']).sum()
print(f"\n5. Duplicate records: {dups}")

# %% [code] Cell 5: Handle missing values
# 2021 is missing entirely — this is likely a COVID data gap
# Strategy: For modeling, we'll note this gap but not fabricate data
# For SARIMA forecasting, we'll use the continuous series excluding 2021

# Fill NaN arrivals with 0 (countries with no data in certain quarters/years)
df_long['arrivals'] = df_long['arrivals'].fillna(0)
df_total['total_arrivals'] = df_total['total_arrivals'].fillna(0)

# Create a complete country-year-quarter grid for analysis
all_countries = sorted(df_long['country'].unique())
all_years = sorted(df_long['year'].unique())
all_quarters = ['Q1', 'Q2', 'Q3', 'Q4']

idx = pd.MultiIndex.from_product([all_countries, all_years, all_quarters],
                                  names=['country', 'year', 'quarter'])
df_complete = df_long.set_index(['country', 'year', 'quarter']).reindex(idx, fill_value=0).reset_index()

print(f"df_complete shape: {df_complete.shape}")
print(f"Zero-valued cells filled: {(df_complete['arrivals'] == 0).sum()}")

# %% [code] Cell 6: Feature engineering
# Add numeric quarter for modeling
quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
df_complete['quarter_num'] = df_complete['quarter'].map(quarter_map)

# Create a continuous time index (year + quarter fraction)
df_complete['time_idx'] = df_complete['year'] + (df_complete['quarter_num'] - 1) / 4

# Lag features for modeling (per country)
df_complete = df_complete.sort_values(['country', 'year', 'quarter_num'])
for lag in [1, 2, 4]:
    df_complete[f'lag_{lag}'] = df_complete.groupby('country')['arrivals'].shift(lag)

# Rolling mean (4 quarters = 1 year)
df_complete['rolling_mean_4'] = (df_complete.groupby('country')['arrivals']
                                  .transform(lambda x: x.rolling(4, min_periods=1).mean()))

# YoY growth
df_complete['yoy_growth'] = df_complete.groupby(['country', 'quarter_num'])['arrivals'].pct_change()

print("Feature engineering complete.")
print(f"Shape: {df_complete.shape}")
print(f"Columns: {list(df_complete.columns)}")
print(f"\nSample (Hoa Kỳ, last 8 quarters):")
print(df_complete[df_complete['country'] == 'Hoa Kỳ'].tail(8).to_string())

# %% [code] Cell 7: Train-test split
# Use years <= 2023 for training, 2024+ for testing
# Skip 2021 gap for time series models
TRAIN_END = 2023
TEST_START = 2024

train_mask = df_complete['year'] <= TRAIN_END
test_mask = df_complete['year'] >= TEST_START

df_train = df_complete[train_mask].copy()
df_test = df_complete[test_mask].copy()

print(f"Training set: {df_train.shape[0]} rows, years {df_train['year'].min()}-{df_train['year'].max()}")
print(f"Test set:     {df_test.shape[0]} rows, years {df_test['year'].min()}-{df_test['year'].max()}")

# For aggregate forecasting (total arrivals across all countries)
total_train = df_total[df_total['year'] <= TRAIN_END].copy()
total_test = df_total[df_total['year'] >= TEST_START].copy()

print(f"\nTotal arrivals training: {len(total_train)} rows")
print(f"Total arrivals test: {len(total_test)} rows")

# %% [code] Cell 8: EDA — Total arrivals trend
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

# Annotate COVID dip
ax.annotate('COVID-19\nImpact', xy=(2020, yearly_total[yearly_total['year']==2020]['total_arrivals'].values[0]/1e6),
            xytext=(2018.5, 3), fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_total_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: eda_total_trend.png")

# %% [code] Cell 9: EDA — Top 10 countries bar chart
top10_countries = (df_total.groupby('country')['total_arrivals'].sum()
                   .sort_values(ascending=False).head(10))

fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette('viridis', len(top10_countries))
bars = ax.barh(range(len(top10_countries)), top10_countries.values / 1e6, color=colors)
ax.set_yticks(range(len(top10_countries)))
ax.set_yticklabels(top10_countries.index, fontsize=11)
ax.set_xlabel('Total Arrivals (millions)', fontsize=12)
ax.set_title('Top 10 Source Countries — Total Arrivals (All Years)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, top10_countries.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val/1e6:.1f}M', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('eda_top10_countries.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: eda_top10_countries.png")

# %% [code] Cell 10: EDA — Seasonality analysis
fig, ax = plt.subplots(figsize=(12, 6))
quarterly = df_long.groupby('quarter')['arrivals'].agg(['mean', 'std']).reindex(['Q1', 'Q2', 'Q3', 'Q4'])

ax.bar(quarterly.index, quarterly['mean'] / 1e3, yerr=quarterly['std'] / 1e3,
       capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], edgecolor='black')
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Mean Arrivals (thousands)', fontsize=12)
ax.set_title('Seasonality: Average Arrivals by Quarter', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: eda_seasonality.png")

# %% [code] Cell 11: EDA — Correlation heatmap (top countries)
top5 = df_total.groupby('country')['total_arrivals'].sum().sort_values(ascending=False).head(5).index.tolist()
pivot = df_total[df_total['country'].isin(top5)].pivot_table(
    index='year', columns='country', values='total_arrivals', aggfunc='sum')
corr = pivot.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Correlation Between Top 5 Source Countries', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: eda_correlation.png")

# %% [code] Cell 12: EDA — Country-specific trends (top 5)
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for i, country in enumerate(top5):
    cdata = df_total[df_total['country'] == country].sort_values('year')
    axes[i].plot(cdata['year'], cdata['total_arrivals'] / 1e3, 'o-', linewidth=2, markersize=6)
    axes[i].set_title(country, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Arrivals (thousands)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks(cdata['year'][::2])

# Remove empty subplot
axes[5].set_visible(False)
plt.suptitle('Country-Specific Arrival Trends', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_country_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: eda_country_trend