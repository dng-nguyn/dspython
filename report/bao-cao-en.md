---
title: "Analysis and Forecasting of Vietnam's International Tourist Arrivals Using Monthly Data"
subtitle: "Data Analysis with Python — Final Report"
author: "Nguyen Dinh Cap (dng-nguyn)"
date: "June 2026"
---

\newpage

# Table of Contents

2. Introduction
3. Literature Review
4. Data Collection and Parsing
5. Data Preprocessing
6. Exploratory Data Analysis
7. Model Building
8. Model Optimization and External Features
9. Forecasting and Evaluation
10. Conclusion
11. References

\newpage

# Introduction

## Background

Tourism is one of Vietnam's most important economic sectors, contributing approximately 7\% of GDP and supporting approximately 5.96 million jobs as of 2024 [1]. Vietnam welcomed a record 18.0 million international visitors in 2019, ranking fifth in the Asia-Pacific region (fourth in Southeast Asia according to PATA H1 2019 data) [2,26]. The COVID-19 pandemic caused international arrivals to fall by 78.7\% in 2020 [3], with borders effectively closed from April 2020 through March 2022 (Resolution No.~32/NQ-CP). Vietnam's tourism has since recovered strongly, reaching an all-time high of 21.2 million arrivals in 2025 [4].

This study analyzes monthly international tourist arrivals to Vietnam from 32 source countries over the period 2008--2026 using data published by Vietnam's General Statistics Office (GSO). Monthly granularity provides approximately 12 times as many observations as quarterly aggregation, enabling more robust statistical inference and finer seasonal resolution.

## Objectives

1. Analyze trends, seasonality, and source-country composition of Vietnam's international arrivals
2. Compare forecasting models: Linear Regression, Random Forest, XGBoost, SARIMAX, Chronos-T5 foundation model, and the CIR\# stochastic differential equation model
3. Evaluate the impact of external features (exchange rates, visa policy) on forecast accuracy
4. Generate 12-month-ahead forecasts with confidence intervals
5. Document model assumptions, limitations, and boundary conditions

## Scope

- **Subject:** Monthly international tourist arrivals to Vietnam by source country
- **Period:** January 2008 -- May 2026 (no data for 2021 due to COVID-19 border closures; only January--March 2020 available)
- **Countries:** 32 individual countries (regional aggregates excluded)
- **Tools:** Python 3, pandas, scikit-learn, XGBoost, statsmodels, Chronos-T5, yfinance

\newpage

# Literature Review

Tourism demand forecasting has been extensively studied. Song and Witt [5] provide a foundational treatment of econometric tourism demand modelling. A comprehensive review by Song et al. [6], covering 211 papers, finds that no single method dominates all contexts; performance depends on data granularity, forecast horizon, and structural breaks.

For time-series methods, Hyndman and Athanasopoulos [7] present the standard reference for ARIMA-family models, including seasonal extensions (SARIMA). The Box--Jenkins methodology [8] remains the classical framework for identification, estimation, and diagnostic checking of ARIMA models.

Ensemble machine learning methods have gained prominence. Random Forest [9] reduces variance through bagging of decorrelated trees. XGBoost [10] builds trees sequentially with gradient boosting and regularization, excelling at capturing nonlinear feature interactions but prone to overfitting on small datasets. The bias-variance tradeoff [11] explains why simpler models can outperform complex ones when sample sizes are limited.

Foundation models represent a recent paradigm shift. Chronos [12], developed by Amazon, pre-trains Transformer-based models on millions of time series for zero-shot forecasting without task-specific training.

Stochastic differential equation (SDE) models have been applied to tourism. Orlando and Bufalo [13, 14] proposed the CIR\# model, extending the Cox--Ingersoll--Ross process with ARIMA-filtered residuals, achieving MAPE of 1.18\% on Italian monthly tourism data (288 observations). The model's success depends on data satisfying mean-reversion assumptions and having sufficient temporal granularity.

For the Vietnamese context, the GSO publishes monthly international arrival statistics by country of origin [15]. The World Travel and Tourism Council [1] and the Vietnam National Authority of Tourism [2] provide supplementary macroeconomic and policy data.

Missing data in tourism series is a well-known challenge. Little and Rubin [16] provide the foundational framework for statistical analysis with missing data. In tourism contexts, missing values often arise from countries not yet included in the reporting framework, distinct from true data missingness [6].

\newpage

# Data Collection and Parsing

## Data Source

The data consists of 12 HTML-Excel report files (`t1.xls` through `t12.xls`), one per calendar month, downloaded from the GSO website (`https://www.gso.gov.vn/`) in June 2026. Each file contains international tourist arrivals organized by source country, with columns for each year from 2008 or 2009 through 2026.

## Parsing Challenges

The `.xls` files are HTML documents with an `.xls` extension, not standard Excel binaries. The HTML structure presents several challenges:

1. **Text nodes outside `<td>` tags.** Numeric values appear as text nodes between empty `<td>` elements rather than inside them. Standard parsers such as `pd.read_html()` return NaN for numeric values that appear outside proper \texttt{<td>} elements. A custom `lxml`-based parser was constructed.

2. **Inconsistent year coverage.** Months t7 and t9--t12 include 2008 data; t1--t6 and t8 start from 2009. Coverage for early years (2009--2011) is limited to 11--13 countries.

3. **Vietnamese number formatting.** Dots serve as thousands separators (e.g., `33.379` equals 33,379).

4. **Regional aggregates.** Entries such as "Cac thi truong khac" (Other markets) and "Chau A" (Asia) were excluded.

## Parsing Verification

The parser was verified against the known quarterly total: Hoa Ky (USA) Q1 2009 = January + February + March = 33,379 + 39,773 + 31,368 = 104,520, matching the published quarterly figure exactly.

## Parsed Dataset Summary

| Metric | Value |
|--------|-------|
| Total records | 4,692 |
| Countries | 32 |
| Date range | July 2008 -- May 2026 |
| Missing year | 2021 (border closures) |
| Incomplete year | 2020 (January--March only) |

## Country Coverage Analysis

Country coverage is highly uneven across the time period:

| Period | Countries per month | Note |
|--------|-------------------|------|
| 2008 (Jul--Dec) | 30 | t7, t9--t12 only |
| 2009--2011 | 11--13 | Severely limited |
| 2012--2017 | 29--30 | Stable coverage |
| 2018--2019 | 31 | Slight expansion |
| 2020 (Jan--Mar) | 31 | Pre-COVID only |
| 2022--2026 | 29--31 | Post-COVID recovery |

This coverage gap means that aggregate totals for 2009--2011 are artificially low. All trend analyses in this report use the period 2012 onward, where coverage is stable at 29--31 countries.

\newpage

# Data Preprocessing

## Cleaning

- **Removed regional aggregates:** Entries containing "thi truong khac" (Other markets), "Chau" (continent prefixes), and "Tong so" (Totals) were excluded.
- **Removed the "Totals" row** that appeared as a data row in some files.
- **Country names** kept in Vietnamese for analysis; English translations used in this report.
- **No duplicate records** were found.

## Handling Missing Values

- **2020--2021 COVID closure:** For aggregate monthly analysis, April 2020--December 2021 are zero-imputed (arrivals = 0) to preserve calendar continuity. Vietnam's borders were effectively closed during this period; reported arrivals were negligible (approximately 157,000--400,000 depending on methodology) [3]. A binary `covid_closed` exogenous variable (1 for these months, 0 otherwise) is added to signal the structural break to SARIMAX.
- **Sparse countries:** Ba Lan (Poland, 29 months of data starting from January 2024), Cong hoa Sec (Czech Republic, 1 month in 2026), and An Do (India, 67 months from 2018) have limited data. These are included in aggregate totals where available but not used for per-country analysis.
- **Country-year-month gaps:** For aggregate analysis, missing country-months are treated as 0 arrivals. This implicitly assumes arrivals were negligible for countries that entered the reporting framework later (e.g., India from 2018, Poland from 2024). While this assumption holds for the aggregate COVID-period imputation, it may understate arrivals for certain countries in specific years [16].

## Train-Test Split

- **Training set:** January 2012 -- December 2023 (144 months, continuous). Includes zero-imputed COVID closure months (April 2020--December 2021) with `covid_closed = 1`, providing the model with a complete calendar and the structural break signal.
- **Test set:** January 2024 -- December 2025 (24 months). Full post-COVID recovery period.
- **Excluded from training:** 2008--2011 (limited country coverage to 11--13 countries).
- **Forecast horizon:** January -- December 2026 (12 months ahead).

This yields 144 training observations and 24 test observations for monthly aggregate analysis, compared to 51 and 10 in the previous quarterly analysis. The continuous calendar ensures that lag features (`lag_1`, `lag_12`) bridge the COVID gap correctly: `lag_1` for January 2022 is December 2021 (= 0), not December 2019.

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `year` | Calendar year |
| `month` | Month (1--12) |
| `time_idx` | Year + (month$-$1)/12, continuous time index |
| `lag_1` | Previous month's total arrivals |
| `lag_12` | Same month in previous year |
| `rolling_mean_12` | 12-month trailing average |
| `covid_closed` | Binary indicator (1 for April 2020--December 2021, 0 otherwise) |
| `exchange_rate_*` | VND spot rates vs. 8 currencies (optional, lagged by 1 month) |
| `visa_*` | Visa policy indicators (optional) |
Exchange rates were obtained from Yahoo Finance as end-of-month spot rates [17]. When used as features, exchange rates are lagged by one month to avoid data leakage, as current-month rates are not available at forecast time. Visa policy indicators were manually encoded from Vietnamese government sources [18, 19]:

- `visa_evisa`: Binary, set to 1 from February 2017 (pilot e-visa for 40 countries)
- `visa_evisa_full`: Binary, set to 1 from August 15, 2023 (universal 90-day e-visa)
- `covid_restrict`: Ordinal (0--1), encoding travel restriction severity

\newpage

# Exploratory Data Analysis

## Overall Trend with Coverage Context

![Total annual arrivals with country-count overlay. Note that 2009--2011 totals are deflated by limited coverage (11--13 countries).](output/eda_total_trend.png)

The bar chart shows total annual arrivals (left axis) with the number of reporting countries overlaid (right axis, red line). Key observations:

- **2009--2011:** Only 11--13 countries report, making totals artificially low
- **2012--2019:** Stable growth with 29--31 countries reporting; arrivals grew from 6.8 million (2012) to 18.0 million (2019)
- **2020:** Only January--March data available; borders closed from April
- **2022--2025:** Strong post-COVID recovery, reaching 21.2 million in 2025 [4]

## Top Source Countries

![Top 10 source countries by cumulative arrivals (2008--2026).](output/eda_top10_countries.png)

China (cumulative 41.7 million) and South Korea (33.0 million) are the dominant source markets. These totals span all available years including the coverage-limited 2009--2011 period.

## Growth Rates (2012--2019)

![Average annual growth rate by country (2012--2019).](output/eda_growth_rate.png)

Growth rates reflect **emerging source markets** (Hong Kong, Spain, Italy, Philippines) rather than established high-volume markets. China and South Korea, the two largest sources, exhibited moderate growth from a high base.

## Monthly Seasonality

![Monthly seasonality pattern across all years and countries (2012--2019, 2022--2025).](output/eda_seasonality.png)

- **January--February** has the highest arrivals, driven by the Tet holiday (Lunar New Year) and winter tourism
- **June--August** shows a secondary peak (summer tourism)
- **November--December** has the lowest arrivals on average

## Correlation Between Source Markets

![Correlation matrix of top 5 source countries (monthly arrivals).](output/eda_correlation.png)

China and Taiwan exhibit the highest correlation (approximately 0.89), reflecting shared geography and similar travel patterns. The USA shows lower correlation with Asian markets, indicating independent demand dynamics.

## Country-Specific Trends

![Monthly arrivals for top 5 source countries (2008--2026).](output/eda_country_trends.png)

- **South Korea** recovered fastest post-COVID
- **China** suffered the steepest decline and slowest recovery
- **Japan** shows steady but moderate growth throughout the period

\newpage
# Model Building

This section describes each forecasting model applied to **aggregate monthly arrivals** (summed across all countries). Every model uses the feature set described in Section 4.3. Models are evaluated on the test set (January 2024 -- December 2025, 24 months) using four metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R$^2$** (Coefficient of Determination): Proportion of variance explained relative to the training-set mean. Negative values indicate performance worse than predicting the training-set mean, which is expected when the test distribution (post-COVID recovery) differs fundamentally from the training distribution (pre-COVID era) [11].

## Linear Regression

Linear Regression assumes a linear relationship between features $\mathbf{x}$ and target $y$:

$$y = \mathbf{w}^T \mathbf{x} + b$$

The model minimizes the sum of squared residuals (Ordinary Least Squares):

$$\min_{\mathbf{w}, b} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

**Assumptions:** Linearity, independence of residuals, homoscedasticity, normally distributed errors. With 144 training samples and 7 features (including intercept), the observation-to-parameter ratio (144:8 $\approx$ 18:1) is adequate.

**Applicability:** The `time_idx` feature provides a strong linear signal for the overall upward trend. With 144 training samples, the model's parsimony is an advantage over complex alternatives [11].

**Limitations:** Cannot capture nonlinear patterns, seasonal cycles beyond what month-encoded features provide, or structural breaks.

## Random Forest Regression

Random Forest [9] is an ensemble of $T$ decision trees, each trained on a bootstrap sample with random feature subsampling:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

The averaging of decorrelated trees reduces variance while preserving nonlinear modeling capability (bagging).

**Assumptions:** i.i.d. samples. Does not require stationarity or linearity.
**Key hyperparameters:** `n_estimators` = 200, `max_depth` = 10 (to limit overfitting on 144 samples), `min_samples_split` = 2.

**Advantage over quarterly analysis:** With 144 training samples (vs. 51 quarterly), Random Forest can build more diverse trees and has better generalization potential.

## XGBoost Regressor

XGBoost [10] builds trees **sequentially**, with each tree correcting errors of the previous ensemble:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

At each step, a new tree minimizes a regularized objective:

$$\mathcal{L}^{(k)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)) + \Omega(f_k)$$

where $\ell$ is squared error and $\Omega$ penalizes tree complexity.

**Assumptions:** i.i.d. samples. Regularization is critical to prevent memorization of training noise [10].

**Key hyperparameters:** `n_estimators` = 200, `max_depth` = 6, `learning_rate` = 0.1.

## SARIMAX$(1,1,1)(1,1,1)_{12}$

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) [7, 8] is a classical statistical model decomposing a time series into autoregressive, differencing, moving-average, and seasonal components, augmented with exogenous regressors:

$$\Phi_P(B^s)\,\phi_p(B)\,(1-B^s)^D\,(1-B)^d\, y_t = \sum_{j} \beta_j x_{j,t} + \Theta_Q(B^s)\,\theta_q(B)\,\varepsilon_t$$

where $B$ is the backshift operator ($B y_t = y_{t-1}$), $s = 12$ is the seasonal period, $x_{j,t}$ are exogenous variables, and $\varepsilon_t$ is white noise.

The model used here, SARIMAX$(1,1,1)(1,1,1)_{12}$, applies first-order differencing at both regular and seasonal levels, with one autoregressive and one moving-average term at each level. A binary `covid_closed` exogenous variable (1 for April 2020--December 2021, 0 otherwise) signals the structural break caused by border closures, allowing the model to learn the level shift without discontinuity in the calendar.

**Assumptions:** Stationarity after differencing; white-noise residuals. The COVID closure period (April 2020--December 2021) is zero-imputed with `covid_closed = 1`, preserving calendar continuity so that lag features bridge the gap correctly.

**Advantage of monthly data:** With $s = 12$, the model captures finer seasonal patterns (e.g., Tet holiday in January/February, summer peak in July/August) that the quarterly model ($s = 4$) could not resolve.

## Chronos-T5

Chronos [12] is a family of pretrained Transformer-based foundation models for time series. Unlike the other models, Chronos does not learn from the 144 training samples; it was pretrained on millions of time series from diverse domains. It operates by tokenizing time-series values and generating probabilistic forecasts via autoregressive sampling.

| Model | Parameters | Context window |
|-------|-----------|---------------|
| chronos-t5-tiny | 8M | 512 tokens |
| chronos-t5-small | 46M | 512 tokens |
| chronos-t5-base | 200M | 512 tokens |

**Advantage of monthly data:** With 144 context points (vs. 55 quarterly), the model has a denser signal for pattern recognition. The 512-token context window is well within limits.

## CIR\# Stochastic Differential Equation Model

The CIR\# model [13, 14] extends the Cox--Ingersoll--Ross SDE for tourism forecasting with disrupted data:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

Orlando and Bufalo [13] report MAPE of 1.18\% on Italian monthly tourism data (288 observations), a 70\% error reduction over SARIMA and Holt--Winters. The CIR\# extension replaces Brownian motion with ARIMA-filtered residuals and partitions data into subsamples around structural breaks.

**Evaluation on monthly Vietnam data:** With 144 training observations, CIR\# has sufficient data for MLE parameter estimation but faces three fundamental challenges: (1) Vietnam's tourism exhibits a strong upward trend that violates the mean-reversion assumption ($\kappa(\theta - r(t))$ term); (2) the 78.7\% COVID drop followed by full recovery creates extreme log-return volatility amplified by the $\sqrt{r}$ diffusion term; (3) the model is designed for stationary, mean-reverting processes [14].

## Model Comparison

![Model performance comparison (MAE, MAPE, R$^2$).](output/model_comparison.png)

| Model              | MAE       | RMSE      | MAPE   | R²       |
|--------------------|-----------|-----------|--------|----------|
| Chronos-T5-small   | 170,625   | 214,069   | 10.77% | −0.0345  |
| Linear Regression  | 275,526   | 340,495   | 19.79% | 0.2439   |
| Random Forest      | 290,287   | 334,896   | 19.80% | 0.2685   |
| XGBoost            | 313,159   | 369,109   | 19.88% | 0.1114   |
| SARIMAX (log)        | 402,014   | 512,715   | 26.87% | −0.7145  |
| CIR#               | 489,200   | 573,449   | 28.54% | −1.1447  |

**Key observations:**

- **Chronos-T5-small achieves the lowest overall error** (MAPE = 10.77\%, MAE = 170,625), demonstrating the strong zero-shot generalization of foundation models on complex, structurally broken datasets. Its negative R$^2$ ($-$0.03) reflects the difficulty of extrapolating the post-COVID growth trend from a model pretrained on general time-series patterns.
- **Linear Regression and tree-based models achieve moderate but positive R$^2$ values** (0.11--0.27) and similar MAPEs ($\approx$ 19.8\%). While they successfully extract some signal from the data, their accuracy is heavily constrained by the structural break between the training period (2012--2023, including zero-imputed COVID months) and the test period (2024--2025 post-COVID recovery).
- **Feature importance** (Random Forest): `lag_1` dominates at 69.8\%, followed by `lag_12` (11.3\%) and `time_idx` (8.7\%). The previous month's arrivals are the strongest predictor. The dominance of `lag_1` means the tree-based models function primarily as naive one-step-ahead forecasters. This works well on the 24-month test set (where each month's actual lag is available) but would degrade significantly in multi-step-ahead forecasting scenarios where lag values must be recursively predicted.
- **SARIMAX** (MAPE = 26.87\%, R$^2$ = $-$0.71) with log-transformed target and `covid_closed` exogenous variable. The log-transformation ensures non-negative confidence intervals, but the model still struggles to extrapolate the post-COVID growth trend from a stationary-process framework.
- **CIR\# fails** (MAPE = 28.54\%, R$^2$ = $-$1.14) despite having monthly data. CIR\# was included in the comparison specifically to empirically validate its documented boundary condition [13, 14]: the model requires mean-reverting stationary processes. The estimated $\kappa$ is positive (mean-reverting), which conflicts with the upward-trending data. This confirms the model's documented boundary condition [13, 14].

\newpage

# Model Optimization and External Features

## Hyperparameter Optimization

**Random Forest** was optimized via GridSearchCV (3-fold cross-validation):

- `n_estimators`: {100, 200, 300}
- `max_depth`: {5, 10, 15, None}
- `min_samples_split`: {2, 5, 10}

**XGBoost** was optimized via RandomizedSearchCV (50 iterations):

- `n_estimators`: 100--500
- `max_depth`: 3--9
- `learning_rate`: 0.01--0.2
- `subsample`: 0.7--1.0
- `colsample_bytree`: 0.7--1.0

## External Features

Exchange rates (VND vs. KRW, CNY, USD, JPY, TWD, MYR, THB, RUB) were obtained from Yahoo Finance [17] as end-of-month spot rates. Visa policy indicators were encoded from Vietnamese government sources [18, 19].

**Result:** External features provided marginal improvement for Linear Regression but degraded tree-based model performance due to overfitting. Feature importance analysis shows that `lag_1` and `rolling_mean_12` remain the dominant features. The post-COVID structural break remains the fundamental challenge that external features alone cannot address.

\newpage

# Forecasting and Evaluation

## Predicted vs. Actual (Test Set)

![Predicted vs. actual monthly arrivals for the test set (2024--2025).](output/pred_vs_actual.png)

**Forecasting methodology note.** All four models generate 12-month-ahead forecasts for 2026. The tree-based models (Linear Regression, Random Forest, XGBoost) use a recursive strategy: each month's prediction is fed back as the `lag_1` feature for the next month. This accumulates error at each step but captures the nonlinear patterns the tree models learned. SARIMAX uses its autoregressive structure to generate multi-step predictions directly. The ensemble mean (shown in red) averages all four models, providing a more robust forecast than any single model. The shaded band shows the range between the most optimistic and most pessimistic model — a more interpretable uncertainty measure than the SARIMAX confidence interval, which spans several orders of magnitude due to the log-transformation. Furthermore, dynamic external features (such as exchange rates) were excluded from the final out-of-sample 2026 models to prevent data leakage, meaning the multi-step forecasts rely strictly on autoregressive patterns, calendar indices, and policy indicators.

## 12-Month Ensemble Forecast (2026)
![SARIMAX 12-month forecast for 2026 with 95\% confidence interval.](output/forecast_plot.png)

| Month | Forecast | 95\% CI Lower | 95\% CI Upper |
|-------|----------|--------------|--------------|
| Jan 2026 | 1,538,229 | 940,658 | 2,515,418 |
| Feb 2026 | 1,334,457 | 700,080 | 2,543,676 |
| Mar 2026 | 1,077,130 | 491,657 | 2,359,791 |
| Apr 2026 | 1,164,874 | 475,096 | 2,856,116 |
| May 2026 | 1,074,181 | 395,703 | 2,915,977 |
| Jun 2026 | 1,400,343 | 470,662 | 4,166,384 |
| Jul 2026 | 1,546,696 | 477,592 | 5,009,008 |
| Aug 2026 | 1,713,283 | 488,855 | 6,004,510 |
| Sep 2026 | 1,586,391 | 420,225 | 5,988,773 |
| Oct 2026 | 1,601,608 | 395,427 | 6,487,018 |
| Nov 2026 | 1,436,842 | 331,758 | 6,222,945 |
| Dec 2026 | 1,854,630 | 401,648 | 8,563,820 |

To prevent the forecasting of physically impossible negative arrivals and to stabilize variance, the target variable was log-transformed ($\log(y+1)$) prior to SARIMAX fitting. The resulting forecasts and confidence intervals were exponentiated back to the original scale using $\exp(\cdot) - 1$, naturally bounding the lower confidence intervals at zero without the need for arbitrary manual clipping. This approach produces wider confidence intervals than the raw-scale model, reflecting the asymmetric uncertainty inherent in multiplicative processes. The extreme width of the confidence intervals (e.g., December 2026 upper bound of 8,563,820 vs. point estimate 1,854,630) may also indicate model misspecification for multi-step forecasting with log-transformed targets, rather than merely reflecting genuine uncertainty.

## Per-Country Forecasts (2026)

Ensemble models (averaging Linear Regression, Random Forest, XGBoost, and SARIMAX) were fitted to the top 5 source countries individually. The tree-based models use recursive forecasting to generate 12-month-ahead predictions; SARIMAX uses its autoregressive structure directly.

![Ensemble forecasts for top 5 source countries (2026). Model range band shows disagreement between LR, RF, XGBoost, and SARIMAX.](output/country_forecasts_plot.png)

| Month | Hàn Quốc | Trung Quốc | Campuchia | Nhật Bản | Nga |
|-------|----------|------------|-----------|----------|-----|
| Jan 2026 | 389,721 | 212,894 | 32,843 | 60,466 | 14,745 |
| Feb 2026 | 407,507 | 230,786 | 31,082 | 60,571 | 15,503 |
| Mar 2026 | 363,183 | 226,083 | 31,519 | 62,357 | 14,771 |
| Apr 2026 | 350,430 | 227,242 | 30,196 | 59,207 | 13,652 |
| May 2026 | 354,600 | 245,202 | 29,337 | 60,160 | 13,218 |
| Jun 2026 | 363,588 | 252,661 | 28,884 | 58,926 | 12,304 |
| Jul 2026 | 371,555 | 265,062 | 30,280 | 60,739 | 11,455 |
| Aug 2026 | 420,453 | 277,285 | 29,165 | 75,738 | 11,884 |
| Sep 2026 | 400,238 | 269,141 | 32,144 | 78,329 | 12,094 |
| Oct 2026 | 406,438 | 271,010 | 32,529 | 64,637 | 12,410 |
| Nov 2026 | 411,074 | 288,898 | 34,186 | 66,296 | 14,932 |
| Dec 2026 | 429,562 | 309,729 | 34,750 | 70,569 | 15,117 |
| **Total** | **4,668,349** | **3,075,993** | **376,915** | **777,995** | **162,085** |

*Table shows ensemble mean of Linear Regression, Random Forest, XGBoost, and SARIMAX. Model disagreement range: 8.2M–10.1M total.*

The top 5 countries account for 52.3\% of the total 2026 ensemble forecast (9.1M of 17.3M). Hàn Quốc is projected to remain the largest source market, followed by Trung Quốc. The shaded band shows the range between the highest and lowest model predictions — a more interpretable measure of uncertainty than the SARIMAX confidence interval.

\newpage

## Forecast Validation (Jan–May 2026)
With actual data available for the first five months of 2026, we can evaluate the SARIMAX-only forecast against reality (the ensemble forecast uses the same SARIMAX component for its aggregate prediction).

![Forecast vs Actual for Jan—May 2026 (aggregate and top source countries).](output/forecast_validation.png)

| Month | Actual | Forecast | Error |
|-------|--------|----------|-------|
| Jan 2026 | 1,641,403 | 1,169,591 | −28.7% |
| Feb 2026 | 2,124,123 | 1,225,010 | −42.3% |
| Mar 2026 | 1,540,586 | 1,077,132 | −30.1% |
| Apr 2026 | 1,601,269 | 1,164,874 | −27.3% |
| May 2026 | 1,553,853 | 1,074,181 | −30.8% |
| **MAPE** | | | **34.7%** |

**Per-country validation:**

| Country | MAPE | Notes |
|---------|------|-------|
| Hàn Quốc | 8.3% | Best fit; model captures seasonal pattern well |
| Trung Quốc | 48.1% | Consistent underprediction; structural growth since 2024 not captured |
| Nhật Bản | 25.9% | Improved with corrected parser; seasonal pattern partially captured |
| Campuchia | 50.3% | Land-border regime shift: visa exemptions + new air routes + healthcare tourism; Jan 2026 hit 223K (3× Dec 2025) |

**Nga (Russia) case study.** The SARIMAX component forecasts 7,000--12,000 arrivals/month for Nga in 2026 (the ensemble mean in the table is higher because the other three models pull the average up); actual figures are 113,000--137,000 — a 10$\times$ error. This is not a model failure but a geopolitical regime shift. The Russia-Ukraine war (2022) closed EU and US destinations to Russian tourists. Vietnam, with its 45-day visa exemption, resumed direct flights (Aeroflot, VietJet, Vietnam Airlines), and affordable pricing, became a primary alternative. Russian arrivals grew from 39,921 (2022, per the GSO monthly dataset; the VNAT-published annual total for 2022 is 28,056) to 689,714 (2025), surpassing the pre-pandemic peak of 646,524 (2019). No model trained on 2012--2023 data could anticipate this structural redirection of Russian outbound tourism [20].

**Campuchia case study.** The model forecasts 27,000–38,000 Cambodian arrivals/month for 2026; actual figures range from 53,000 to 223,000. Two effects compound:

1. **Baseline regime shift.** Cambodia's monthly average rose from ~38,000 (2024) to ~57,000 (2025–2026), driven by reciprocal 30-day visa exemptions, Air Cambodia's new routes (~10 daily flights to Vietnam), and accelerating cross-border healthcare tourism and shopping. Cambodia's proximity (6-hour bus from Phnom Penh to Ho Chi Minh City) makes it fundamentally different from air-dependent markets.

2. **Tet seasonal amplification.** January spikes recur: January 2025 hit 100,000 (+64\% over December); January 2026 hit 223,025 (+206\% over December). Cambodians cross the border en masse for Lunar New Year shopping and family visits. The 2026 spike was double the 2025 spike due to new flight capacity launched in late 2025. February–May 2026 settled to 53,000–66,000/month, confirming the elevated baseline.

The model, trained on 2012–2023 data showing Cambodia at 20,000–70,000/month, cannot anticipate either effect [21].

The aggregate MAPE of 34.7% confirms that the SARIMAX model systematically underestimates post-COVID growth acceleration. Hàn Quốc is the only country well-predicted (MAPE = 8.3\%), as its growth trajectory most closely resembles the 2012–2023 training distribution. The Campuchia case illustrates a fundamental limitation: a model trained on pre-2024 data cannot anticipate a 10× regime shift.

# Conclusion

## Key Findings

1. **Monthly data significantly improves analysis.** With 144 training observations (vs. 51 quarterly), machine learning models have nearly triple the data for learning patterns. SARIMAX with $s = 12$ resolves finer seasonal structure (Tet holiday, summer peaks) that $s = 4$ could not capture.

2. **Coverage bias persists.** Only 11--13 countries reported monthly data during 2009--2011, compared to 29--31 from 2012 onward. Aggregate trend analyses should use 2012 as the starting point.

3. **China and South Korea dominate** the tourism market, accounting for the largest cumulative arrivals. Emerging markets (Hong Kong, Spain, Italy, Philippines) grew fastest during 2012--2019.

4. **Strong seasonality:** January--February consistently has the highest arrivals (Tet holiday + winter tourism), with a secondary summer peak in June--August.

5. **Post-COVID structural break** remains the dominant challenge. Three models (Random Forest, Linear Regression, XGBoost) achieve positive R$^2$ values (0.11--0.27), while Chronos, SARIMAX, and CIR\# remain negative ($-$0.03, $-$0.71, $-$1.14 respectively) because the test distribution (2024--2025 recovery) differs fundamentally from the training distribution (2012--2023).

6. **CIR\# fails on trending data.** Despite having monthly data as recommended by Orlando and Bufalo [13], the model's mean-reversion assumption is violated by Vietnam's upward-trending tourism trajectory. This constitutes a documented boundary condition for the model [13, 14].

7. **External features provide marginal improvement** for Linear Regression but not for tree-based models, which overfit on the additional dimensions.

## Limitations

1. **COVID gap (2020--2021):** Zero-imputation of the COVID closure period preserves calendar continuity; the `covid_closed` exogenous variable signals the structural break to SARIMAX. However, the zero-filled months still represent a departure from the true data-generating process.
2. **Limited training set (144 months):** While substantially better than 51 quarterly points, this still constrains model complexity compared to the 288 monthly observations used in the Italian CIR\# study [13].
3. **Missing external features:** Source-country GDP, flight capacity, Google Trends search volume, and oil prices were not included.
4. **Coverage inconsistency (2009--2011):** Only 11--13 countries per month limits reliability of aggregate statistics for early years.
5. **Formal diagnostics not performed:** Residual normality tests, stationarity tests (ADF, KPSS), and heteroscedasticity tests would strengthen the analysis.

## Future Directions

1. **Source-country GDP and flight capacity** as leading indicators
2. **Google Trends search volume** for destination queries
3. **Regime-switching models** that explicitly handle pre-COVID, during-COVID, and post-COVID transitions
4. **Per-country models** (different countries exhibit different recovery trajectories)
5. **Ensemble methods** combining Linear Regression (trend) with Chronos (pattern recognition)
6. **Formal residual diagnostics** and stationarity testing

\newpage

# References

[1] World Travel and Tourism Council, "Economic Impact Research: Vietnam," WTTC, 2024. [Online]. Available: https://wttc.org/research/economic-impact

[2] Vietnam National Authority of Tourism, "International tourist arrivals to Vietnam 2019," VNAT, 2020. Cited in: B-Company, "Vietnam Tourism Briefing," Jan. 2025.

[3] H. T. Thi, "Vietnam Tourism Industry During Covid-19 Pandemic," 2022. See also: General Statistics Office of Vietnam, "Socio-economic situation report 2020," GSO, Hanoi, 2021.

[4] VietnamNet and General Statistics Office of Vietnam, "Vietnam achieved 21.2 million international arrivals in 2025," VietnamNet, Jan. 2026. [Online]. Available: https://vietnamnet.vn/

[5] H. Song and S. F. Witt, *Tourism Demand Modelling and Forecasting: Modern Econometric Approaches*. Routledge, 2000.

[6] H. Song, G. Li, and Z. Cao, "Tourism demand modelling and forecasting --- A review of recent research," *Tourism Management*, vol. 74, pp. 217--232, 2019.

[7] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. [Online]. Available: https://otexts.com/fpp3/

[8] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2016.

[9] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5--32, 2001.

[10] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785--794.

[11] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.

[12] A. F. Ansari et al., "Chronos: Learning the Language of Time Series," *Trans. Mach. Learn. Res.*, 2024. [Online]. Available: https://arxiv.org/abs/2403.07815

[13] G. Orlando and M. Bufalo, "Improved tourism demand forecasting with CIR\# model: the case of disrupted data patterns in Italy," *Tourism Review*, vol. 79, no. 2, pp. 445--470, 2023, doi: 10.1108/TR-04-2023-0194.

[14] G. Orlando and M. Bufalo, "The CIR\# model for time series forecasting," *Technological and Economic Development of Economy*, vol. 29, no. 5, pp. 1403--1427, 2023.

[15] General Statistics Office of Vietnam, "Quarterly and monthly international arrival statistics by country," GSO, Hanoi, 2008--May 2026 (partial). [Online]. Available: https://www.gso.gov.vn/

[16] R. J. A. Little and D. B. Rubin, *Statistical Analysis with Missing Data*, 2nd ed. Wiley, 2002.

[17] Yahoo Finance, "Historical exchange rates," Yahoo, 2025. [Online]. Available: https://finance.yahoo.com/

[18] Vietnam Immigration Department, "Pilot e-visa system for foreign visitors," effective Feb. 1, 2017, initially for 40 countries.

[19] Vietnam National Assembly, "Resolution on extension and amendment of e-visa policy," approved Jun. 24, 2023, effective Aug. 15, 2023.

[20] VietnamPlus, "Russian tourists to Vietnam surge in 2024," Vietnam News Agency, 2024. [Online]. Available: https://en.vietnamplus.vn/

[21] VnExpress International, "Cambodia replaces Taiwan to become Vietnam's 3rd largest source of tourists," Feb. 2026.

[22] Scikit-learn developers, "Scikit-learn: Machine Learning in Python," 2025. [Online]. Available: https://scikit-learn.org/

[23] T. Chen et al., "XGBoost Documentation," 2025. [Online]. Available: https://xgboost.readthedocs.io/

[24] Statsmodels developers, "Statsmodels: Statistical Modeling and Econometrics in Python," 2025. [Online]. Available: https://www.statsmodels.org/

[25] Amazon Science, "Chronos Forecasting," GitHub, 2024. [Online]. Available: https://github.com/amazon-science/chronos-forecasting

[26] Pacific Asia Travel Association, "Visitor Arrivals to Asia Pacific Destinations," PATA, H1 2019.
