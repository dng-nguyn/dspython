---
title: "Analysis and Forecasting of Vietnam's International Tourist Arrivals Using Monthly Data"
subtitle: "Data Analysis with Python — Final Report"
author: "Nguyen Dinh Cap (dng-nguyn)"
date: "July 2025"
---

\newpage

# Table of Contents

1. Introduction
2. Literature Review
3. Data Collection and Parsing
4. Data Preprocessing
5. Exploratory Data Analysis
6. Model Building
7. Model Optimization and External Features
8. Forecasting and Evaluation
9. Conclusion
10. References

\newpage

# Introduction

## Background

Tourism is one of Vietnam's most important economic sectors, contributing approximately 6--7\% of GDP and supporting nearly 6 million jobs nationally in recent years [1]. Vietnam welcomed a record 18.0 million international visitors in 2019, ranking fourth in Southeast Asia [2]. The COVID-19 pandemic caused international arrivals to fall by 78.7\% in 2020 [3], with borders effectively closed from April 2020 through early 2022. Vietnam's tourism has since recovered strongly, reaching an all-time high of 21.2 million arrivals in 2025 [4].

This study analyzes monthly international tourist arrivals to Vietnam from 12 source countries over the period 2008--2026 using data published by Vietnam's General Statistics Office (GSO). Monthly granularity provides approximately 12$\times$ more observations than quarterly aggregation, enabling more robust statistical inference and finer seasonal resolution.

## Objectives

1. Analyze trends, seasonality, and source-country composition of Vietnam's international arrivals
2. Compare forecasting models: Linear Regression, Random Forest, XGBoost, SARIMA, Chronos-T5 foundation model, and the CIR\# stochastic differential equation model
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

Stochastic differential equation (SDE) models have been applied to tourism. Orlando and Buffa [13, 14] proposed the CIR\# model, extending the Cox--Ingersoll--Ross process with ARIMA-filtered residuals, achieving MAPE of 1.18\% on Italian monthly tourism data (288 observations). The model's success depends on data satisfying mean-reversion assumptions and having sufficient temporal granularity.

For the Vietnamese context, the GSO publishes monthly international arrival statistics by country of origin [15]. The World Travel and Tourism Council [1] and the Vietnam National Authority of Tourism [2] provide supplementary macroeconomic and policy data.

Missing data in tourism series is a well-known challenge. Little and Rubin [16] provide the foundational framework for statistical analysis with missing data. In tourism contexts, missing values often arise from countries not yet included in the reporting framework, distinct from true data missingness [6].

\newpage

# Data Collection and Parsing

## Data Source

The data consists of 12 HTML-Excel report files (`t1.xls` through `t12.xls`), one per calendar month, downloaded from the GSO website (`https://www.gso.gov.vn/`) in July 2025. Each file contains international tourist arrivals organized by source country, with columns for each year from 2008 or 2009 through 2026.

## Parsing Challenges

The `.xls` files are HTML documents with an `.xls` extension, not standard Excel binaries. The HTML structure presents several challenges:

1. **Text nodes outside `<td>` tags.** Numeric values appear as text nodes between empty `<td>` elements rather than inside them. Standard parsers such as `pd.read_html()` return all NaN values. A custom `lxml`-based parser was constructed.

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

- **2021:** Excluded entirely. Vietnam's borders were closed; reported arrivals were negligible (approximately 157,000--400,000 depending on methodology) [3].
- **2020:** Only January--March data available. April--December excluded (border closures).
- **Sparse countries:** Ba Lan (Poland, 29 months from 2024), Cong hoa Sec (Czech Republic, 1 month in 2026), and An Do (India, 67 months from 2018) have limited data. These are included in aggregate totals where available but not used for per-country analysis.
- **Country-year-month gaps:** For aggregate analysis, missing country-months are treated as 0 arrivals. This is justified because countries absent from the reporting framework in a given period had negligible arrivals [16].

## Train-Test Split

- **Training set:** January 2012 -- December 2019 + January 2022 -- December 2023 (120 months). Combines stable pre-COVID coverage (29--31 countries) with early post-COVID recovery data.
- **Test set:** January 2024 -- December 2025 (24 months). Full post-COVID recovery period.
- **Excluded from training:** 2008--2011 (limited country coverage), 2020--2021 (COVID gap: only January--March 2020 available, 2021 entirely absent).
- **Forecast horizon:** January -- December 2026 (12 months ahead).

This yields 120 training observations and 24 test observations for monthly aggregate analysis, compared to 51 and 10 in the previous quarterly analysis. Including 2022--2023 in the training set provides the model with early recovery patterns, improving its ability to extrapolate the growth trajectory.

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `year` | Calendar year |
| `month` | Month (1--12) |
| `time_idx` | Year + (month$-$1)/12, continuous time index |
| `lag_1` | Previous month's total arrivals |
| `lag_12` | Same month in previous year |
| `rolling_mean_12` | 12-month trailing average |
| `exchange_rate_*` | VND spot rates vs. 8 currencies (optional) |
| `visa_*` | Visa policy indicators (optional) |

Exchange rates were obtained from Yahoo Finance as end-of-month spot rates [17]. Visa policy indicators were manually encoded from Vietnamese government sources [18, 19]:

- `visa_evisa`: Binary, set to 1 from January 2017 (pilot e-visa for 40 countries)
- `visa_evisa_full`: Binary, set to 1 from August 2023 (universal 90-day e-visa)
- `covid_restrict`: Ordinal (0--1), encoding travel restriction severity

\newpage

# Exploratory Data Analysis

## Overall Trend with Coverage Context

![Figure 1: Total annual arrivals with country-count overlay. Note that 2009--2011 totals are deflated by limited coverage (11--13 countries).](output/eda_total_trend.png)

The bar chart shows total annual arrivals (left axis) with the number of reporting countries overlaid (right axis, red line). Key observations:

- **2009--2011:** Only 11--13 countries report, making totals artificially low
- **2012--2019:** Stable growth with 29--31 countries reporting; arrivals grew from 6.8 million (2012) to 18.0 million (2019)
- **2020:** Only January--March data available; borders closed from April
- **2022--2025:** Strong post-COVID recovery, reaching 21.2 million in 2025 [4]

## Top Source Countries

![Figure 2: Top 10 source countries by cumulative arrivals (2008--2026).](output/eda_top10_countries.png)

China (cumulative 41.7 million) and South Korea (33.0 million) are the dominant source markets. These totals span all available years including the coverage-limited 2009--2011 period.

## Growth Rates (2012--2019)

![Figure 3: Average annual growth rate by country (2012--2019).](output/eda_growth_rate.png)

Growth rates reflect **emerging source markets** (Hong Kong, Spain, Italy, Philippines) rather than established high-volume markets. China and South Korea, the two largest sources, exhibited moderate growth from a high base.

## Monthly Seasonality

![Figure 4: Monthly seasonality pattern across all years and countries (2012--2019, 2022--2025).](output/eda_seasonality.png)

- **January--February** has the highest arrivals, driven by the Tet holiday (Lunar New Year) and winter tourism
- **June--August** shows a secondary peak (summer tourism)
- **November--December** has the lowest arrivals on average

## Correlation Between Source Markets

![Figure 5: Correlation matrix of top 5 source countries (monthly arrivals).](output/eda_correlation.png)

China and Taiwan exhibit the highest correlation (approximately 0.89), reflecting shared geography and similar travel patterns. The USA shows lower correlation with Asian markets, indicating independent demand dynamics.

## Country-Specific Trends

![Figure 6: Monthly arrivals for top 5 source countries (2008--2026).](output/eda_country_trends.png)

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

**Assumptions:** Linearity, independence of residuals, homoscedasticity, normally distributed errors. With 120 training samples and 6 features, the observation-to-parameter ratio (24:1) is adequate.

**Applicability:** The `time_idx` feature provides a strong linear signal for the overall upward trend. With 120 training samples, the model's parsimony is an advantage over complex alternatives [11].

**Limitations:** Cannot capture nonlinear patterns, seasonal cycles beyond what month-encoded features provide, or structural breaks.

## Random Forest Regression

Random Forest [9] is an ensemble of $T$ decision trees, each trained on a bootstrap sample with random feature subsampling:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

The averaging of decorrelated trees reduces variance while preserving nonlinear modeling capability (bagging).

**Assumptions:** i.i.d. samples. Does not require stationarity or linearity.
**Key hyperparameters:** `n_estimators` = 200, `max_depth` = 10 (to limit overfitting on 120 samples), `min_samples_split` = 2.

**Advantage over quarterly analysis:** With 120 training samples (vs. 51 quarterly), Random Forest can build more diverse trees and has better generalization potential.

## XGBoost Regressor

XGBoost [10] builds trees **sequentially**, with each tree correcting errors of the previous ensemble:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

At each step, a new tree minimizes a regularized objective:

$$\mathcal{L}^{(k)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)) + \Omega(f_k)$$

where $\ell$ is squared error and $\Omega$ penalizes tree complexity.

**Assumptions:** i.i.d. samples. Regularization is critical to prevent memorization of training noise [10].

**Key hyperparameters:** `n_estimators` = 200, `max_depth` = 6, `learning_rate` = 0.1.

## SARIMA$(1,1,1)(1,1,1)_{12}$

SARIMA (Seasonal AutoRegressive Integrated Moving Average) [7, 8] is a classical statistical model decomposing a time series into autoregressive, differencing, moving-average, and seasonal components:

$$\Phi_P(B^s)\,\phi_p(B)\,(1-B^s)^D\,(1-B)^d\, y_t = \Theta_Q(B^s)\,\theta_q(B)\,\varepsilon_t$$

where $B$ is the backshift operator ($B y_t = y_{t-1}$), $s = 12$ is the seasonal period, and $\varepsilon_t$ is white noise.

The model used here, SARIMA$(1,1,1)(1,1,1)_{12}$, applies first-order differencing at both regular and seasonal levels, with one autoregressive and one moving-average term at each level.

**Assumptions:** Stationarity after differencing; white-noise residuals. The 2020--2021 data gap breaks seasonal continuity, as the model observes March 2020 followed directly by January 2022.

**Advantage of monthly data:** With $s = 12$, the model captures finer seasonal patterns (e.g., Tet holiday in January/February, summer peak in July/August) that the quarterly model ($s = 4$) could not resolve.

## Chronos-T5

Chronos [12] is a family of pretrained Transformer-based foundation models for time series. Unlike the other models, Chronos does not learn from the 120 training samples; it was pretrained on millions of time series from diverse domains. It operates by tokenizing time-series values and generating probabilistic forecasts via autoregressive sampling.

| Model | Parameters | Context window |
|-------|-----------|---------------|
| chronos-t5-tiny | 8M | 512 tokens |
| chronos-t5-small | 46M | 512 tokens |
| chronos-t5-base | 200M | 512 tokens |

**Advantage of monthly data:** With 120 context points (vs. 55 quarterly), the model has a denser signal for pattern recognition. The 512-token context window is well within limits.

## CIR\# Stochastic Differential Equation Model

The CIR\# model [13, 14] extends the Cox--Ingersoll--Ross SDE for tourism forecasting with disrupted data:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

Orlando and Buffa [13] report MAPE of 1.18\% on Italian monthly tourism data (288 observations), a 70\% error reduction over SARIMA and Holt--Winters. The CIR\# extension replaces Brownian motion with ARIMA-filtered residuals and partitions data into subsamples around structural breaks.

**Evaluation on monthly Vietnam data:** With 120 training observations, CIR\# has sufficient data for MLE parameter estimation but faces three fundamental challenges: (1) Vietnam's tourism exhibits a strong upward trend that violates the mean-reversion assumption ($\kappa(\theta - r(t))$ term); (2) the 80\% COVID drop followed by full recovery creates extreme log-return volatility amplified by the $\sqrt{r}$ diffusion term; (3) the model is designed for stationary, mean-reverting processes [14].

## Model Comparison

![Figure 7: Model performance comparison (MAE, MAPE, R$^2$).](output/model_comparison.png)

| Model | MAE | RMSE | MAPE | R$^2$ |
|-------|-----|------|------|-------|
| XGBoost | 112,072 | 138,597 | 7.47\% | 0.5663 |
| Random Forest | 121,491 | 141,631 | 8.21\% | 0.5471 |
| Linear Regression | 133,072 | 158,772 | 9.16\% | 0.4309 |
| Chronos-T5-small | 170,625 | 214,069 | 10.77\% | $-$0.0345 |
| CIR\# | 511,410 | 575,996 | 32.91\% | $-$6.4900 |
| SARIMA$(1,1,1)(1,1,1)_{12}$ | 712,583 | 738,893 | 47.77\% | $-$11.3255 |

**Key observations:**

- **XGBoost achieves the best performance** (MAPE = 7.47\%, R$^2$ = 0.57), followed closely by Random Forest (8.21\%) and Linear Regression (9.16\%).
- **Tree-based models now have positive R$^2$ values**, a dramatic improvement over the quarterly analysis where all R$^2$ were negative. This confirms that monthly data with 120 training observations provides substantially more information than 51 quarterly points.
- **Feature importance** (Random Forest): `lag_1` dominates at 84.8\%, followed by `lag_12` (6.4\%) and `rolling_mean_12` (5.0\%). The previous month's arrivals are the strongest predictor.
- **Chronos-T5-small** performs competitively (MAPE = 10.77\%) as a zero-shot method but has negative R$^2$, indicating it does not fully capture the post-COVID growth trend.
- **SARIMA performs worst** (MAPE = 47.77\%, R$^2$ = $-$11.33). The 2020--2021 data gap breaks seasonal continuity; the model forecasts a mean-reverting pattern rather than the observed growth trajectory.
- **CIR\# fails** (MAPE = 32.91\%, R$^2$ = $-$6.49) despite having monthly data. The estimated $\kappa = 0.109$ is positive (mean-reverting), which conflicts with the upward-trending data. This confirms the model's documented boundary condition [13, 14].

\newpage

# Model Optimization and External Features

## Hyperparameter Optimization

**Random Forest** was optimized via GridSearchCV (3-fold cross-validation):

- `n_estimators`: {100, 200, 300}
- `max_depth`: {5, 10, 15, None}
- `min_samples\_split`: {2, 5, 10}

**XGBoost** was optimized via RandomizedSearchCV (50 iterations):

- `n\_estimators`: 100--500
- `max\_depth`: 3--9
- `learning\_rate`: 0.01--0.2
- `subsample`: 0.7--1.0
- `colsample\_bytree`: 0.7--1.0

## External Features

Exchange rates (VND vs. KRW, CNY, USD, JPY, TWD, MYR, THB, RUB) were obtained from Yahoo Finance [17] as end-of-month spot rates. Visa policy indicators were encoded from Vietnamese government sources [18, 19].

**Result:** External features provided marginal improvement for Linear Regression but degraded tree-based model performance due to overfitting. Feature importance analysis shows that `lag_1` and `rolling_mean_12` remain the dominant features. The post-COVID structural break remains the fundamental challenge that external features alone cannot address.

\newpage

# Forecasting and Evaluation

## Predicted vs. Actual (Test Set)

![Figure 8: Predicted vs. actual monthly arrivals for the test set (2024--2025).](output/pred_vs_actual.png)

XGBoost and Random Forest track the actual values most closely, while SARIMA and CIR\# diverge significantly. Chronos-T5-small provides competitive zero-shot predictions without any training on this dataset.

## 12-Month Forecast (2026)

![Figure 9: SARIMA 12-month forecast for 2026 with 95\% confidence interval.](output/forecast_plot.png)

| Month | Forecast | 95\% CI Lower | 95\% CI Upper |
|-------|----------|--------------|--------------|
| Jan 2026 | 633,903 | 121,322 | 1,146,483 |
| Feb 2026 | 676,032 | $-$83,685 | 1,435,750 |
| Mar 2026 | 607,735 | $-$326,054 | 1,541,524 |
| Apr 2026 | 649,047 | $-$434,436 | 1,732,532 |
| May 2026 | 607,630 | $-$606,144 | 1,821,405 |
| Jun 2026 | 597,289 | $-$734,454 | 1,929,033 |
| Jul 2026 | 689,660 | $-$750,295 | 2,129,616 |
| Aug 2026 | 829,177 | $-$711,451 | 2,369,806 |
| Sep 2026 | 775,281 | $-$859,818 | 2,410,381 |
| Oct 2026 | 808,583 | $-$915,825 | 2,532,992 |
| Nov 2026 | 921,835 | $-$887,477 | 2,731,148 |
| Dec 2026 | 961,391 | $-$929,016 | 2,851,799 |

The wide confidence intervals (some extending below zero) reflect fundamental uncertainty. SARIMA, having been trained on pre-COVID data, forecasts a mean-reverting pattern centered around 600,000--900,000 monthly arrivals, well below the observed 2025 levels (1.5--2.0 million per month). This underscores the limitation of classical time-series models in the presence of structural breaks.

\newpage

# Conclusion

## Key Findings

1. **Monthly data significantly improves analysis.** With 120 training observations (vs. 51 quarterly), machine learning models have nearly triple the data for learning patterns. SARIMA with $s = 12$ resolves finer seasonal structure (Tet holiday, summer peaks) that $s = 4$ could not capture.

2. **Coverage bias persists.** Only 11--13 countries reported monthly data during 2009--2011, compared to 29--31 from 2012 onward. Aggregate trend analyses should use 2012 as the starting point.

3. **China and South Korea dominate** the tourism market, accounting for the largest cumulative arrivals. Emerging markets (Hong Kong, Spain, Italy, Philippines) grew fastest during 2012--2019.

4. **Strong seasonality:** January--February consistently has the highest arrivals (Tet holiday + winter tourism), with a secondary summer peak in June--August.

5. **Post-COVID structural break** remains the dominant challenge. All models produce negative R$^2$ values because the test distribution (2024--2025 recovery) differs fundamentally from the training distribution (2012--2019 + 2022--2023).

6. **CIR\# fails on trending data.** Despite having monthly data as recommended by Orlando and Buffa [13], the model's mean-reversion assumption is violated by Vietnam's upward-trending tourism trajectory. This constitutes a documented boundary condition for the model [13, 14].

7. **External features provide marginal improvement** for Linear Regression but not for tree-based models, which overfit on the additional dimensions.

## Limitations

1. **COVID gap (2020--2021):** Excluding 20 months breaks time-series continuity for SARIMA and reduces the training set.
2. **Limited training set (120 months):** While substantially better than 51 quarterly points, this still constrains model complexity compared to the 288 monthly observations used in the Italian CIR\# study [13].
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

[5] H. Song and S. F. Witt, *Tourism Demand Modelling and Forecasting: Modern Econometric Approaches*. Routledge, 2012.

[6] H. Song, G. Li, and Z. Cao, "Tourism demand modelling and forecasting --- A review of recent research," *Tourism Management*, vol. 74, pp. 217--232, 2019.

[7] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. [Online]. Available: https://otexts.com/fpp3/

[8] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2016.

[9] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5--32, 2001.

[10] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785--794.

[11] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.

[12] A. F. Ansari et al., "Chronos: Learning the Language of Time Series," *Trans. Mach. Learn. Res.*, 2024. [Online]. Available: https://arxiv.org/abs/2403.07815

[13] G. Orlando and M. Buffa, "Improved tourism demand forecasting with CIR\# model: the case of disrupted data patterns in Italy," *Tourism Review*, vol. 79, no. 2, pp. 445--470, 2023, doi: 10.1108/TR-04-2023-0194.

[14] G. Orlando and M. Buffa, "The CIR\# model for time series forecasting," *Technological and Economic Development of Economy*, vol. 29, no. 5, pp. 1403--1427, 2023.

[15] General Statistics Office of Vietnam, "Quarterly and monthly international arrival statistics by country," GSO, Hanoi, 2008--2026. [Online]. Available: https://www.gso.gov.vn/

[16] R. J. A. Little and D. B. Rubin, *Statistical Analysis with Missing Data*, 2nd ed. Wiley, 2002.

[17] Yahoo Finance, "Historical exchange rates," Yahoo, 2025. [Online]. Available: https://finance.yahoo.com/

[18] Vietnam Immigration Department, "Pilot e-visa system for foreign visitors," effective Feb. 1, 2017, initially for 40 countries.

[19] Vietnam National Assembly, "Resolution on extension and amendment of e-visa policy," approved Jun. 24, 2023, effective Aug. 15, 2023.

[20] Scikit-learn developers, "Scikit-learn: Machine Learning in Python," 2025. [Online]. Available: https://scikit-learn.org/

[21] T. Chen et al., "XGBoost Documentation," 2025. [Online]. Available: https://xgboost.readthedocs.io/

[22] Statsmodels developers, "Statsmodels: Statistical Modeling and Econometrics in Python," 2025. [Online]. Available: https://www.statsmodels.org/

[23] Amazon Science, "Chronos Forecasting," GitHub, 2024. [Online]. Available: https://github.com/amazon-science/chronos-forecasting
