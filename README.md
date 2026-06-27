# Phân tích Du lịch Quốc tế Việt Nam (2008–2026)

Phân tích và dự đoán lượng khách quốc tế đến Việt Nam theo quốc gia nguồn và quý, sử dụng các mô hình học máy và foundation model.

## Cấu trúc thư mục

```
dspython/
├── data/              # Dữ liệu thô (XLS files)
├── output/            # CSV, PNG đầu ra
├── notebooks/         # Jupyter notebook phân tích
├── scripts/           # Script Python tái sử dụng
└── report/            # Báo cáo (MD, TEX, PDF)
```

## Cách chạy

### 1. Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn lxml scikit-learn xgboost statsmodels chronos-forecasting torch
```

### 2. Mở notebook

```bash
cd notebooks/
jupyter notebook bao-cao.ipynb
```

### 3. Chạy từ đầu

Cell đầu tiên (Section 0) sẽ tự động tải dữ liệu và tạo thư mục:

```python
# Tải dữ liệu từ GSO
!wget -q https://files.catbox.moe/83h5ir.zip
!mkdir -p data output
!unzip -q -o 83h5ir.zip -d data
!rm 83h5ir.zip
```

Sau đó **Run All** để chạy toàn bộ pipeline.

### 4. Script trực tiếp (không cần Jupyter)

```bash
python scripts/analysis.py
```

### 5. Tạo lại báo cáo

```bash
python scripts/create_report.py
# → Xuất ra report/bao-cao.md, report/bao-cao.tex, report/bao-cao.pdf
```

## Output

| File | Mô tả |
|------|-------|
| `output/df_long.csv` | Dữ liệu dài (country × year × quarter) |
| `output/df_total.csv` | Tổng khách theo năm và quốc gia |
| `output/model_comparison.csv` | So sánh 7 mô hình |
| `output/forecast.csv` | Dự đoán 4 quý tiếp theo |
| `output/eda_*.png` | Biểu đồ EDA |
| `output/model_comparison.png` | So sánh hiệu suất |
| `output/pred_vs_actual.png` | Dự đoán vs thực tế |
| `output/forecast_plot.png` | Biểu đồ dự đoán |

## Mô hình sử dụng

| Mô hình | Loại | Ghi chú |
|---------|------|---------|
| Linear Regression | Hồi quy tuyến tính | Baseline đơn giản |
| Random Forest | Ensemble (trees) | GridSearchCV tối ưu |
| XGBoost | Gradient boosting | RandomizedSearchCV tối ưu |
| SARIMA | Chuỗi thời gian | Mùa vụ quarterly |
| Chronos-T5 | Foundation model | Zero-shot, pretrained trên hàng triệu chuỗi |

## Dữ liệu nguồn

- 4 file HTML-Excel từ Tổng cục Thống kê (GSO)
- Lượng khách quốc tế theo 40 quốc gia, quý Q1–Q4, 2008–2026
- Năm 2021 không có dữ liệu (COVID-19)
