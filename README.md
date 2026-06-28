# Phân tích Du lịch Quốc tế Việt Nam (2008–2026)

Phân tích và dự đoán lượng khách quốc tế đến Việt Nam theo quốc gia nguồn và tháng, sử dụng các mô hình học máy, SARIMA, Chronos-T5 foundation model, và CIR# stochastic model.

## Cấu trúc thư mục

```
dspython/
├── data/              # Dữ liệu thô (t1.xls – t12.xls + quarterly files)
├── output/            # CSV, PNG đầu ra
├── notebooks/         # Jupyter notebook phân tích
├── scripts/           # Script Python tái sử dụng
└── report/            # Báo cáo (MD, TEX, PDF)
```

## Cách chạy

### 1. Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn lxml scikit-learn xgboost statsmodels yfinance
# Optional (for Chronos-T5):
pip install chronos-forecasting torch
```

### 2. Chạy notebook

```bash
cd notebooks/
jupyter notebook bao-cao.ipynb
```

### 3. Tạo báo cáo PDF

```bash
python3 scripts/create_report.py
# → Xuất ra report/bao-cao.pdf (Vietnamese) và report/bao-cao-en.pdf (English)
```

### 4. Tạo lại notebook từ script

```bash
python3 scripts/create_notebook.py
```

## Output

| File | Mô tả |
|------|-------|
| `output/df_monthly.csv` | Dữ liệu hàng tháng (country × year × month) |
| `output/model_results.csv` | So sánh 6 mô hình |
| `output/forecast.csv` | Dự báo 12 tháng (2026) |
| `output/pred_vs_actual.csv` | Dự đoán vs thực tế theo tháng |
| `output/eda_*.png` | 6 biểu đồ EDA |
| `output/model_comparison.png` | So sánh hiệu suất |
| `output/pred_vs_actual.png` | Dự đoán vs thực tế |
| `output/forecast_plot.png` | Biểu đồ dự báo |

## Mô hình sử dụng

| Mô hình | Loại | Ghi chú |
|---------|------|---------|
| Linear Regression | Hồi quy tuyến tính | Baseline, MAPE=9.16% |
| Random Forest | Ensemble (trees) | MAPE=8.21% |
| XGBoost | Gradient boosting | **MAPE=7.47% (tốt nhất)** |
| SARIMA$(1,1,1)(1,1,1)_{12}$ | Chuỗi thời gian | Mùa vụ monthly, MAPE=47.77% |
| Chronos-T5-small | Foundation model | Zero-shot, MAPE=10.77% |
| CIR# | SDE model | Mean-reversion fails, MAPE=32.91% |

## Dữ liệu nguồn

- 12 file HTML-Excel (`t1.xls` – `t12.xls`) từ Tổng cục Thống kê (GSO)
- Lượng khách quốc tế theo 32 quốc gia, hàng tháng, 2008–2026
- Năm 2021 không có dữ liệu (COVID-19 border closures)
- Train: 2012–2019 + 2022–2023 (120 tháng), Test: 2024–2025 (24 tháng)

## Báo cáo

- `report/bao-cao-en.md` / `.pdf` — English report with IEEE citations
- `report/bao-cao.md` / `.pdf` — Vietnamese report
