---
title: "Analysis and Forecasting of Vietnam's International Tourist Arrivals"
subtitle: "Final Report — Data Analysis with Python"
author: "dng-nguyn"
date: "July 2025"
---

\newpage

# Table of Contents

1. Introduction
2. Data Collection
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Model Building
6. Model Optimization
7. Future Forecasting
8. Conclusion

\newpage

# 1. Introduction

## 1.1 Problem Statement

Tourism is one of Vietnam's most important economic sectors, contributing significantly to GDP and creating millions of jobs. In recent years, Vietnam has emerged as an attractive destination in Southeast Asia, drawing large numbers of international tourists from diverse countries.

However, international tourist arrivals are influenced by complex factors: global economic conditions, visa policies, exchange rates, regional competition, and exogenous shocks such as the COVID-19 pandemic (2020-2021). Understanding trends, seasonal patterns, and relationships between source markets is essential for effective tourism strategy planning.

A critical challenge in this dataset is **uneven country coverage across years**: only 12 countries reported data during 2009-2011, compared to 30+ from 2012 onward. This means raw aggregate totals are partially inflated by reporting coverage changes rather than actual growth. This report addresses this issue explicitly.

## 1.2 Objectives

**Analytical objectives:**
- Analyze overall trends in international arrivals to Vietnam (2008-2026)
- Identify top source countries and their changing rankings
- Detect seasonal patterns in quarterly arrivals
- Analyze correlations between source markets
- Assess the impact of COVID-19 on Vietnam's tourism sector
- Evaluate average annual growth rates, accounting for coverage gaps

**Predictive objectives:**
- Build machine learning models to predict quarterly arrivals
- Compare Linear Regression, Random Forest, XGBoost, SARIMA, and Chronos-T5
- Tune hyperparameters to improve model accuracy
- Forecast arrivals for the next 4 quarters with confidence intervals
- Evaluate the CIR# stochastic model (from academic literature) and explain why it does not fit this dataset

## 1.3 Scope and Data Sources

- **Subject:** International tourist arrivals to Vietnam, by source country and quarter (Q1-Q4)
- **Time period:** 2008-2026, noting that 2021 has no data (COVID-19 impact)
- **Countries:** 32 individual countries (excluding regional aggregates like "Other Asian markets")
- **Data source:** Four HTML-Excel reports (`.xls`) from Vietnam's General Statistics Office (GSO)

## 1.4 Tools and Libraries

- **Language:** Python 3
- **Data processing:** pandas, numpy, lxml (HTML parsing)
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, xgboost, statsmodels (SARIMA), chronos-forecasting (Chronos-T5)
- **Environment:** Jupyter Notebook
- **Version control:** Git, GitHub

\newpage

# 2. Data Collection

## 2.1 Data Sources

The data consists of 4 HTML-Excel report files, each containing international tourist arrivals for one quarter:

| File | Quarter | Notes |
|------|---------|-------|
| quy1-cacnuoc.xls | Q1 | Starts from 2009 |
| quy2-cacnuoc.xls | Q2 | Starts from 2009 |
| quy3-cacnuoc.xls | Q3 | Starts from 2008 |
| quy4-cacnuoc.xls | Q4 | Starts from 2008 |

## 2.2 Data Structure and Challenges

The `.xls` files are not real Excel binaries — they are HTML files with an `.xls` extension, a common technique for generating "Excel" files from web applications. The HTML structure presents several challenges:

1. **Text nodes outside `<td>` tags:** Numeric values appear as text between empty `<td>` elements rather than inside them. Standard parsers like `pd.read_html()` return all NaN values. A custom `lxml`-based parser was built to extract data from text nodes.

2. **Inconsistent year coverage:** Q3/Q4 files include 2008 data; Q1/Q2 start from 2009. The year 2021 is absent from all files (COVID-19 lockdown).

3. **Vietnamese number formatting:** Dots serve as thousands separators (e.g., `104.520` = 104,520), not decimal points.

4. **Regional aggregates mixed with countries:** The 2008 file includes entries like "Chau A" (Asia), "Chau Au" (Europe), and "Other markets" alongside individual countries. These were excluded from analysis.

## 2.3 Data Parsing and Merging

A custom HTML parser using `lxml` was built:
- Parse the DOM tree and extract year labels from the header row (columns 2+)
- For each data row, extract country name from `td[0].tail` and values from `td[1:].tail`
- Convert Vietnamese-formatted numbers by removing dot separators
- Merge all 4 files into a single DataFrame with columns: `country`, `year`, `quarter`, `arrivals`

**Verification:** "USA" for 2009 Q1 = 104,520 — confirmed against the provided sample.

After parsing: **1,622 records**, **32 individual countries**, years 2008-2026.

## 2.4 Country Coverage Issue

A critical finding is the uneven number of reporting countries per year:

| Year | Countries Reporting | Note |
|------|-------------------|------|
| 2008 | 29 | Q3/Q4 only (different file structure) |
| 2009-2011 | 11-12 | **Only Q1/Q2 files, severely limited** |
| 2012-2015 | 29-30 | Stable coverage |
| 2016-2026 | 29-35 | Near-complete coverage |

This means **aggregate totals for 2009-2011 are artificially low** because only 11-12 countries are reporting. The apparent "growth" from 2.65M (2009) to 18M (2019) is partially driven by more countries being added to the dataset, not just organic growth.

To address this, this report uses **average annual growth rates** (which are per-country and less affected by coverage) alongside raw totals.

\newpage

# 3. Data Preprocessing

## 3.1 Cleaning

- **Removed regional aggregates:** "Chau A", "Chau Au", "Other markets by continent" entries were excluded. These are summary rows, not individual countries.
- **Removed "Totals" row:** Summary total per year-quarter.
- **No duplicate records found.**
- **Country names kept in Vietnamese** for the notebook; mapped to English for this report.

## 3.2 Handling Missing Values

The 2021 gap (no data at all) was handled by simply excluding 2021 from the time series. For the feature grid, missing country-year-quarter combinations were filled with 0.

## 3.3 Feature Engineering

- **Time features:** `quarter_num` (1-4), `time_idx` (year + quarter/4)
- **Lag features:** `lag_1` (previous quarter), `lag_2`, `lag_4` (same quarter last year)
- **Rolling mean:** 4-quarter rolling average to smooth short-term fluctuations
- **External features:** Exchange rates (Yahoo Finance), visa policy indicators (manually encoded from government sources)

## 3.4 Train-Test Split

- **Training set:** Data up to and including 2023
- **Test set:** 2024 onward (2024 Q1 through 2026 Q2 = 10 quarters)

\newpage

# 4. Exploratory Data Analysis (EDA)

## 4.1 Overall Trend with Coverage Context

![Figure 1: Total arrivals by year with country count overlay](output/eda_total_trend.png)

The bar chart (gray, secondary axis) shows the number of reporting countries per year. Key observations:

- **2009-2011:** Only 11-12 countries report, making totals appear artificially low (~2.6-4.0M)
- **2012+:** Coverage stabilizes at 29-30 countries, enabling fair comparison
- **2020:** Sharp COVID-19 drop to ~3.7M despite full coverage (30 countries)
- **2025:** Recovery to ~17.6M with 29 countries — genuine growth

**Important:** The "growth" from 2009 to 2012 should not be interpreted at face value — it's largely a coverage artifact.

## 4.2 Average Annual Growth Rate

To compare countries fairly regardless of absolute volume, average annual growth rates were computed for 2012-2019 (pre-COVID, stable coverage):

![Figure 2: Top countries by average annual growth rate](output/eda_growth_rate.png)

Growth rates reflect **emerging markets** (Hong Kong, Spain, Italy, Germany, Philippines) rather than established markets. China and South Korea — the two largest markets — grew at more moderate rates from a high base.

## 4.3 Top Source Countries by Volume

![Figure 3: Top 10 source countries (all years combined)](output/eda_top10_countries.png)

China (41.7M cumulative) and South Korea (33.0M) dominate. However, these totals span all years including the coverage-limited 2009-2011 period, so they slightly undercount for countries missing early data.

## 4.4 Seasonality

![Figure 4: Average arrivals by quarter](output/eda_seasonality.png)

- **Q1** has the highest average arrivals (Tet holiday, New Year tourism)
- **Q4** has the highest variance, suggesting year-end tourism depends heavily on specific conditions

## 4.5 Correlation Between Source Markets

![Figure 5: Correlation between top 5 source countries](output/eda_correlation.png)

- **China-Taiwan** correlation: 0.89 (shared geography, similar travel patterns)
- **USA** has low correlation with Asian markets (independent dynamics, affected by USD/VND exchange rate)

## 4.6 Country-Specific Trends

![Figure 6: Trends for top 5 source countries](output/eda_country_trends.png)

- **South Korea** recovered fastest post-COVID
- **China** suffered the steepest drop and slowest recovery
- **Japan** shows steady but slower growth

\newpage

# 5. Model Building

Five models were tested on aggregate quarterly arrivals (all countries summed). All use features: year, quarter number, time index, lag_1, lag_4, rolling_mean_4.

## 5.1 Linear Regression

Linear Regression assumes a linear relationship between features and target. It finds the best-fitting line by minimizing sum of squared errors.

Despite its simplicity, LR captures the overall upward trend effectively.

## 5.2 Random Forest Regression

Random Forest combines many decision trees, each trained on a bootstrap sample. Predictions are averaged across trees, reducing overfitting compared to a single tree.

## 5.3 XGBoost Regressor

XGBoost builds trees sequentially, each correcting the errors of the previous one. Known for strong performance in competitions.

## 5.4 SARIMA (Seasonal ARIMA)

SARIMA combines autoregressive (AR), differencing (I), moving average (MA), and seasonal components. The year 2021 gap was removed from the training series.

## 5.5 Chronos-T5 (Foundation Model)

Chronos is Amazon's pretrained time-series foundation model. It performs **zero-shot forecasting** — no task-specific training needed. Four sizes were tested (Tiny/Small/Base/Large); Base (200M params) gave the best MAE.

## 5.6 CIR# — Stochastic Differential Equation Model (Considered but Not Adopted)

The CIR# (Cox-Ingersoll-Ross Sharp) model from Clements et al. (2023) extends the classic CIR SDE for tourism forecasting with disrupted data. It achieved MAPE 1.18% on Italian monthly tourism data — a 70% error reduction vs SARIMA and Holt-Winters.

**Why CIR# does not fit this dataset:**

1. **Insufficient observations:** Only 55 quarterly data points (vs 288 monthly in the Italian study). The rolling window of M=8 consumes 15% of data.
2. **Mean-reversion assumption violated:** The CIR SDE pulls toward a long-run mean θ. Vietnam's tourism has a strong upward trend that contradicts mean-reversion.
3. **COVID shock magnitude:** An 80% drop followed by full recovery causes the √r volatility term to amplify noise in log-returns, producing wild oscillations.
4. **Quarterly vs monthly granularity:** The original paper's success relied on monthly data providing more observations per window and better seasonal capture.

CIR# achieved MAPE ~183% on this data — far worse than all other models. This is an honest negative result: **not every state-of-the-art model fits every dataset**. CIR# is designed for stationary, mean-reverting processes (interest rates, short-term economic indicators), not for trending series with structural breaks.

## 5.7 Model Comparison

![Figure 7: Model performance comparison](output/model_comparison.png)

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Linear Regression | 668,001 | 826,017 | 14.96% | -0.11 |
| Random Forest | 718,697 | 943,153 | 14.73% | -0.44 |
| Chronos-T5-Base | 760,788 | 970,698 | 15.42% | -0.53 |
| XGBoost | 983,132 | 1,396,363 | 21.32% | -2.16 |
| SARIMA | 2,191,581 | 2,369,273 | 47.23% | -8.10 |

**Key observations:**

- All R² values are **negative**, meaning every model performs worse than predicting the mean. This is expected — the post-COVID surge is a structural break invisible to history-based models.
- **Linear Regression** has the lowest MAE (668K) and best R² (-0.11). Its linear trend captures the overall direction.
- **Chronos-T5-Base** has the lowest MAPE among ML models (15.42%) despite being zero-shot.
- **SARIMA** performs worst due to the 2021 gap breaking the seasonal continuity.
- **XGBoost overfits** with only 51 training samples.

\newpage

# 6. Model Optimization

## 6.1 GridSearchCV for Random Forest

Hyperparameters tuned: `n_estimators` (100, 200, 300), `max_depth` (5, 10, 15, None), `min_samples_split` (2, 5, 10). 3-fold cross-validation.

## 6.2 RandomizedSearchCV for XGBoost

50 random combinations from: `n_estimators` (100-500), `max_depth` (3-9), `learning_rate` (0.01-0.2), `subsample` (0.7-1.0), `colsample_bytree` (0.7-1.0).

**Result:** Optimization provided marginal improvement for RF but **made XGBoost worse** — overfitting on the small training set. This confirms that with 51 samples, simpler models are preferred.

## 6.3 External Features (Exchange Rates + Visa Policy)

Exchange rates (VND vs KRW, CNY, USD, JPY, TWD, MYR) were fetched from Yahoo Finance. Visa policy indicators were manually encoded:
- `visa_evisa`: E-visa available (2017+)
- `visa_evisa_full`: E-visa for all countries, 90-day (2023 Q3+)
- `covid_restrict`: Travel restriction index (0-1)

**Result:** External features improved LR slightly (MAE reduced ~5%) but made RF/XGB worse due to overfitting. `lag_1` and `rolling_mean_4` remain the dominant features (~73% importance). The post-COVID structural break remains the fundamental challenge.

\newpage

# 7. Future Forecasting

![Figure 8: SARIMA forecast for next 4 quarters](output/forecast_plot.png)

| Quarter | Forecast | 95% Confidence Interval |
|---------|----------|------------------------|
| 2026 Q3 | 4,157,874 | [2,764,158 — 5,551,591] |
| 2026 Q4 | 4,156,459 | [2,435,914 — 5,877,004] |
| 2027 Q1 | 4,129,710 | [2,167,419 — 6,092,001] |
| 2027 Q2 | 4,129,570 | [1,962,482 — 6,296,659] |

**Note:** The very wide confidence intervals reflect fundamental uncertainty. SARIMA cannot capture the growth trend visible in recent quarters (6M+ in 2025-2026) and reverts toward the historical mean (~4M).

![Figure 9: Predicted vs Actual (test set detail)](output/pred_vs_actual.png)

**Per-quarter accuracy (best models):**

| Quarter | Actual | Best Model | Prediction | Error |
|---------|--------|-----------|------------|-------|
| 2024 Q2 | 3,893,572 | Chronos | 3,602,994 | -7.5% |
| 2024 Q3 | 3,656,240 | XGBoost | 3,686,331 | +0.8% |
| 2024 Q4 | 4,428,526 | LR | 4,030,040 | -9.0% |
| 2025 Q1 | 5,408,927 | LR | 4,288,777 | -20.7% |
| 2026 Q1 | 6,058,711 | LR | 4,708,762 | -22.3% |

Models perform well on "normal" quarters (Q2-Q3) but severely underestimate the surge quarters (Q1 of 2025-2026). This is the **post-COVID structural break** problem.

\newpage

# 8. Conclusion

## 8.1 Key Findings

1. **Coverage bias is significant:** Only 11-12 countries reported during 2009-2011 vs 30+ from 2012. Raw aggregate totals are misleading; growth rates should be used for fair comparison.

2. **China and South Korea dominate** the tourism market, but emerging markets (Hong Kong, Spain, Italy, Philippines) grew fastest during 2012-2019.

3. **Strong seasonality:** Q1 consistently has the highest arrivals (Tet holiday effect).

4. **Linear Regression is surprisingly effective** with only 51 training samples, achieving the lowest MAE (668K) and best R² (-0.11). More complex models overfit.

5. **No model predicts the post-COVID surge well.** All R² values are negative. The structural break is invisible to history-based models.

6. **CIR# (state-of-the-art SDE model) fails on this data** due to insufficient observations, mean-reversion assumption violation, and quarterly granularity. This is a valuable negative finding.

7. **External features (exchange rates, visa policy) provide marginal improvement** for LR but not for tree-based models with limited data.

## 8.2 Limitations

1. **Small training set** (51 quarterly observations after removing 2021)
2. **2021 gap** breaks time-series continuity
3. **Missing external features:** GDP, flight capacity, Google Trends, oil prices
4. **Country coverage inconsistency** (2009-2011 only 11-12 countries)

## 8.3 Future Directions

1. **Monthly data** to increase sample size and improve seasonal capture
2. **External leading indicators:** Source-country GDP, exchange rates, Google Trends search volume, flight route capacity
3. **Per-country models** instead of aggregate (different recovery trajectories)
4. **Ensemble methods** combining LR (trend) with Chronos (pattern recognition)
5. **Regime-switching models** that explicitly handle the pre-COVID / during-COVID / post-COVID transitions
6. **Interactive dashboard** for real-time tourism monitoring and forecasting

\newpage

# References

1. Clements, A.E., et al. (2024). "Forecasting disrupted tourism demand: the CIR# model." *Tourism Review*, 79(2), 445-470.
2. Clements, A.E., et al. (2023). "The CIR# model for time series forecasting." *Technological and Economic Development of Economy*, 29(5), 1403-1427.
3. Scikit-learn documentation: https://scikit-learn.org/
4. XGBoost documentation: https://xgboost.readthedocs.io/
5. Statsmodels SARIMA: https://www.statsmodels.org/
6. Chronos (Amazon): https://github.com/amazon-science/chronos-forecasting
7. Yahoo Finance (exchange rate data): https://finance.yahoo.com/
8. Vietnam General Statistics Office (GSO): https://www.gso.gov.vn/
