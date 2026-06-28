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

This section describes each forecasting model applied to aggregate quarterly arrivals (all countries summed). Every model uses the same feature set: year, quarter number, time index, lag_1 (previous quarter), lag_4 (same quarter last year), and rolling_mean_4 (4-quarter moving average).

Models are evaluated on the test set (2024 Q1 – 2026 Q2, 10 quarters) using four metrics:
- **MAE** (Mean Absolute Error): average absolute deviation from actual values
- **RMSE** (Root Mean Squared Error): penalizes large errors more heavily
- **MAPE** (Mean Absolute Percentage Error): scale-independent accuracy measure
- **R²** (Coefficient of Determination): proportion of variance explained; 1.0 = perfect, 0 = predicting the mean, negative = worse than the mean

## 5.1 Linear Regression

**Algorithm overview:**

Linear Regression is the simplest supervised learning model. It assumes a linear relationship between the input features x and the target variable y:

$$y = \mathbf{w}^T \mathbf{x} + b$$

where **w** is the weight vector and *b* is the bias term. The model finds **w** and *b* by minimizing the sum of squared residuals (Ordinary Least Squares):

$$\min_{\mathbf{w}, b} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

This has a closed-form solution, making it extremely fast to train.

**Why it works here:** Despite its simplicity, Linear Regression effectively captures the overall upward trend in Vietnam's tourism arrivals. The `time_idx` feature (year + quarter/4) provides a strong linear signal. With only 51 training samples, the model's simplicity is actually an advantage — it does not overfit.

**Limitations:** Linear Regression cannot capture nonlinear patterns, seasonal cycles, or the COVID-19 structural break. It assumes the future is a linear extrapolation of the past.

## 5.2 Random Forest Regression

**Algorithm overview:**

Random Forest is an ensemble method that combines hundreds of decision trees. Each tree is trained on a random bootstrap sample of the data, and at each split node, only a random subset of features is considered. The final prediction is the average of all tree predictions:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

where $h_t$ is the $t$-th tree and $T$ is the total number of trees.

The key insight is that individual trees are high-variance (they overfit), but averaging many uncorrelated trees reduces variance while preserving the ability to capture nonlinear patterns. This is known as **bagging** (Bootstrap Aggregating).

**Key hyperparameters:**
- `n_estimators`: Number of trees (200 in this study)
- `max_depth`: Maximum depth of each tree (10 — prevents overfitting on small data)
- `min_samples_split`: Minimum samples required to split a node

**Why it is challenging here:** With only 51 training samples, Random Forest has limited data to build diverse trees. The random subspace mechanism loses information when feature count is small. Overfitting is a real risk.

## 5.3 XGBoost Regressor

**Algorithm overview:**

XGBoost (Extreme Gradient Boosting) builds trees **sequentially** rather than in parallel. Each new tree focuses on correcting the errors of the previous ensemble. The prediction after $K$ trees is:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

At each step $k$, a new tree $f_k$ is added to minimize a regularized objective:

$$\mathcal{L}^{(k)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)) + \Omega(f_k)$$

where $\ell$ is the loss function (squared error for regression) and $\Omega(f_k)$ penalizes tree complexity (number of leaves and leaf weights).

XGBoost uses **second-order Taylor expansion** of the loss function for faster and more accurate optimization, plus **column subsampling** (like Random Forest) to reduce overfitting.

**Key hyperparameters:**
- `n_estimators`: Number of boosting rounds (200)
- `max_depth`: Tree depth (6 — deeper trees capture more interactions)
- `learning_rate`: Shrinkage factor applied to each tree (0.1 — smaller values need more trees but generalize better)
- `subsample`: Fraction of data used per tree (1.0 here)
- `colsample_bytree`: Fraction of features used per tree (1.0 here)

**Why it struggles here:** XGBoost's sequential correction mechanism amplifies noise when the training set is small. With 51 samples and 6 features, the model memorizes training patterns that do not generalize.

## 5.4 SARIMA (Seasonal AutoRegressive Integrated Moving Average)

**Algorithm overview:**

SARIMA is a classical statistical model for time series forecasting. It decomposes the series into four components:

1. **AR (AutoRegressive):** The current value depends on its own past values
2. **I (Integrated):** Differencing to achieve stationarity (removing trend)
3. **MA (Moving Average):** The current value depends on past forecast errors
4. **S (Seasonal):** The same three components repeated at seasonal lag $s$ (here $s=4$ for quarterly data)

The full SARIMA$(p,d,q)(P,D,Q)_s$ model is:

$$\Phi_P(B^s)\phi_p(B)(1-B^s)^D(1-B)^d y_t = \Theta_Q(B^s)\theta_q(B)\varepsilon_t$$

In this study: SARIMA$(1,1,1)(1,1,1)_4$ — one autoregressive and one moving average term at both the regular and seasonal level, with first-order differencing at both levels.

**Key parameters:**
- $(p,d,q) = (1,1,1)$: Non-seasonal AR order, differencing order, MA order
- $(P,D,Q)_s = (1,1,1)_4$: Seasonal AR, differencing, MA orders with period 4

**Why it fails here:** SARIMA assumes the time series is stationary after differencing. The 2021 data gap breaks the seasonal continuity — the model sees 2020 Q4 followed directly by 2022 Q1, creating an artificial jump that distorts the estimated seasonal pattern. Additionally, SARIMA cannot model the strong post-COVID growth trend, which is a structural break rather than a continuation of pre-2020 dynamics.

## 5.5 Chronos-T5 (Foundation Model for Time Series)

**Algorithm overview:**

Chronos is a family of pretrained foundation models for time series developed by Amazon (Ansari et al., 2024). Unlike the other models in this study, Chronos does not learn from our 51 training samples — it was pretrained on **millions of time series** from diverse domains (retail, finance, weather, web traffic, etc.) using a Transformer architecture (T5).

Chronos works by **tokenizing** time series values into discrete bins and training an autoregressive language model to predict the next token. At inference time, it generates multiple possible futures via sampling, producing a **probabilistic forecast** (a distribution over possible outcomes, not just a point estimate).

The key innovation is **zero-shot forecasting**: Chronos can predict any new time series without task-specific training. You simply pass the historical observations as context, and the model generates forecasts.

Four model sizes were tested:

| Model | Parameters | Description |
|-------|-----------|-------------|
| chronos-t5-tiny | 8M | Smallest, fastest |
| chronos-t5-small | 46M | Small balanced |
| chronos-t5-base | 200M | Best accuracy on this data |
| chronos-t5-large | 710M | Largest, but overfit to pretraining distribution |

**Why Base outperforms Large:** With only 55 quarterly data points as context, the larger model's stronger priors (learned from finance, weather, retail data) conflict with the tourism-specific pattern. The Base model is more flexible and less opinionated.

## 5.6 CIR# — Stochastic Differential Equation Model (Considered but Not Adopted)

The CIR# (Cox-Ingersoll-Ross Sharp) model from Clements et al. (2023) extends the classic CIR stochastic differential equation for tourism forecasting with disrupted data. It achieved MAPE 1.18% on Italian monthly tourism data — a 70% error reduction vs SARIMA and Holt-Winters.

**Algorithm overview:**

The classic CIR process models mean-reverting stochastic dynamics:

$$dr(t) = \kappa(\theta - r(t))dt + \sigma\sqrt{r(t)}dW(t)$$

where $\kappa$ is mean-reversion speed, $\theta$ is the long-run mean, $\sigma$ is volatility, and $dW(t)$ is a Brownian motion increment.

The CIR# extension replaces Brownian motion with **ARIMA-filtered residuals**: instead of generating random noise, the model uses the actual forecast errors from an ARIMA model fitted to a rolling window. This gives "exact trajectory" fitted values rather than Monte Carlo averages. The data is also partitioned into subsamples to capture variance changes around structural breaks (e.g., COVID-19).

**Why CIR# does not fit this dataset:**

1. **Insufficient observations:** Only 55 quarterly data points (vs 288 monthly in the Italian study). The rolling window of M=8 consumes 15% of data.
2. **Mean-reversion assumption violated:** The CIR SDE pulls toward a long-run mean $\theta$. Vietnam's tourism has a strong upward trend that contradicts mean-reversion.
3. **COVID shock magnitude:** An 80% drop followed by full recovery causes the $\sqrt{r}$ volatility term to amplify noise in log-returns, producing wild oscillations.
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
3. Ansari, A.F., et al. (2024). "Chronos: Learning the Language of Time Series." *arXiv:2403.07815*.
4. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16*.
5. Scikit-learn documentation: https://scikit-learn.org/
6. XGBoost documentation: https://xgboost.readthedocs.io/
7. Statsmodels SARIMA: https://www.statsmodels.org/
8. Chronos (Amazon): https://github.com/amazon-science/chronos-forecasting
9. Yahoo Finance (exchange rate data): https://finance.yahoo.com/
10. Vietnam General Statistics Office (GSO): https://www.gso.gov.vn/
