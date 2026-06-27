#!/usr/bin/env python3
"""Generate the comprehensive report as markdown, then convert to tex/pdf via pandoc."""
import subprocess, os

os.makedirs('report', exist_ok=True)

# Read model comparison data
import csv
with open('output/model_comparison.csv') as f:
    reader = csv.DictReader(f)
    models = list(reader)

# Build comparison table
comp_rows = []
for m in models:
    comp_rows.append(f"| {m['Model']} | {float(m['MAE']):,.0f} | {float(m['RMSE']):,.0f} | {float(m['R²']):.4f} |")
comp_table = "\n".join(comp_rows)

# Read forecast data
with open('output/forecast.csv') as f:
    reader = csv.DictReader(f)
    forecasts = list(reader)

fc_rows = []
for fc in forecasts:
    fc_rows.append(f"| {fc['year']} {fc['quarter']} | {float(fc['forecast']):,.0f} | [{float(fc['ci_lower']):,.0f} — {float(fc['ci_upper']):,.0f}] |")
fc_table = "\n".join(fc_rows)

md = f"""---
title: "Phân tích và Dự đoán Lượng Khách Du Lịch Quốc tế Đến Việt Nam"
subtitle: "Báo cáo Cuối Kỳ — Lập trình Phân tích Dữ liệu với Python"
author: "dng-nguyn"
date: "Tháng 7, 2025"
---

\\newpage

# Mục lục

1. Giới thiệu đề tài
2. Thu thập dữ liệu
3. Tiền xử lý dữ liệu
4. Phân tích và khám phá dữ liệu (EDA)
5. Xây dựng mô hình dự đoán
6. Tối ưu mô hình
7. Dự đoán tương lai
8. Tổng kết

\\newpage

# 1. Giới thiệu đề tài

## 1.1 Đặt vấn đề bài toán

Ngành du lịch là một trong những ngành kinh tế quan trọng của Việt Nam, đóng góp đáng kể vào GDP và tạo ra hàng triệu việc làm. Trong những năm gần đây, Việt Nam đã nổi lên như một điểm đến hấp dẫn trong khu vực Đông Nam Á, thu hút lượng lớn khách quốc tế từ nhiều quốc gia khác nhau.

Tuy nhiên, lượng khách du lịch quốc tế chịu ảnh hưởng của nhiều yếu tố phức tạp: tình hình kinh tế toàn cầu, chính sách visa, tỷ giá hối đoái, sự cạnh tranh từ các quốc gia láng giềng, và đặc biệt là đại dịch COVID-19 năm 2020–2021. Việc hiểu rõ các xu hướng, mô hình mùa vụ, và mối quan hệ giữa các thị trường nguồn là điều cần thiết để hoạch định chiến lược phát triển du lịch hiệu quả.

Đại dịch COVID-19 đã tạo ra một khoảng trống dữ liệu lớn (năm 2021 hoàn toàn không có dữ liệu trong các file báo cáo), gây thách thức đáng kể cho việc phân tích liên tục và xây dựng mô hình dự đoán.

## 1.2 Mục tiêu đề tài

**Mục tiêu phân tích (Analytical):**

- Phân tích xu hướng tổng thể lượng khách quốc tế đến Việt Nam trong giai đoạn 2008–2026.
- Xác định các quốc gia nguồn khách hàng đầu và sự thay đổi thứ hạng qua các năm.
- Phát hiện các mô hình mùa vụ (seasonality) trong lượng khách theo quý.
- Phân tích mối tương quan giữa các thị trường nguồn khách.
- Đánh giá tác động của đại dịch COVID-19 đến ngành du lịch Việt Nam.

**Mục tiêu dự đoán (Predictive):**

- Xây dựng các mô hình học máy để dự đoán lượng khách quốc tế theo quý.
- So sánh hiệu quả giữa các phương pháp: Linear Regression, Random Forest, XGBoost, SARIMA và Chronos-T5.
- Tối ưu hóa siêu tham số để cải thiện độ chính xác của mô hình.
- Dự đoán lượng khách cho 4 quý tiếp theo kèm theo khoảng tin cậy.

## 1.3 Phạm vi và nguồn dữ liệu

- **Đối tượng phân tích:** Lượng khách quốc tế đến Việt Nam, phân theo quốc gia nguồn và theo quý (Q1–Q4).
- **Thời gian:** Giai đoạn 2008–2026, với lưu ý năm 2021 không có dữ liệu (ảnh hưởng của đại dịch COVID-19).
- **Quốc gia nguồn:** 40 quốc gia/vùng lãnh thổ.
- **Nguồn dữ liệu:** Bốn file báo cáo dạng HTML-Excel (`.xls`) từ Tổng cục Thống kê.

## 1.4 Công cụ và thư viện sử dụng

- **Ngôn ngữ lập trình:** Python 3
- **Xử lý dữ liệu:** pandas, numpy, lxml (phân tích HTML)
- **Trực quan hóa:** matplotlib, seaborn
- **Xây dựng mô hình:** scikit-learn, xgboost, statsmodels (SARIMA), chronos-forecasting (Chronos-T5)
- **Môi trường:** Jupyter Notebook
- **Quản lý phiên bản:** Git, GitHub

\\newpage

# 2. Thu thập dữ liệu

## 2.1 Nguồn dữ liệu

Dữ liệu được cung cấp dưới dạng 4 file báo cáo HTML-Excel, mỗi file chứa thông tin lượng khách quốc tế đến Việt Nam cho một quý cụ thể:

| File | Quý | Số quốc gia | Ghi chú |
|------|-----|-------------|---------|
| quy1-cacnuoc.xls | Q1 | 38 | Bắt đầu từ năm 2009 |
| quy2-cacnuoc.xls | Q2 | 39 | Bắt đầu từ năm 2009 |
| quy3-cacnuoc.xls | Q3 | 40 | Bắt đầu từ năm 2008 |
| quy4-cacnuoc.xls | Q4 | 38 | Bắt đầu từ năm 2008 |

## 2.2 Cấu trúc dữ liệu và thách thức

Các file `.xls` này không phải là file Excel nhị phân thực sự, mà là file HTML có phần mở rộng `.xls` — một kỹ thuật phổ biến để tạo file "Excel" từ web. Cấu trúc HTML có một số đặc điểm gây thách thức:

**Thứ nhất**, dữ liệu không nằm trong các thẻ `<td>` theo cách thông thường. Các giá trị số liệu nằm dưới dạng text nodes giữa các thẻ `<td>` rỗng. Điều này khiến `pd.read_html()` không thể trích xuất chính xác — cần xây dựng bộ phân tích tùy chỉnh bằng `lxml`.

**Thứ hai**, các file không đồng nhất: Q3 và Q4 có dữ liệu từ 2008, Q1/Q2 chỉ từ 2009. Cột năm 2021 hoàn toàn vắng mặt.

**Thứ ba**, định dạng số sử dụng dấu chấm làm phân cách hàng nghìn (ví dụ: `104.520` = 104,520).

## 2.3 Cách xử lý

Nhóm xây dựng bộ phân tích HTML tùy chỉnh bằng `lxml`: phân tích cây DOM, trích xuất text nodes từ thẻ `<td>`, chuyển đổi định dạng số, và hợp nhất thành DataFrame duy nhất.

**Xác minh:** Quốc gia "Hoa Kỳ" năm 2009, quý 1 = 104,520 lượt khách — khớp với dữ liệu mẫu.

Sau hợp nhất: **1,894 bản ghi**, **40 quốc gia**, **18 năm** (2008–2026, không bao gồm 2021).

\\newpage

# 3. Tiền xử lý dữ liệu

## 3.1 Làm sạch dữ liệu

- **Loại bỏ hàng "Totals"** — hàng tổng cộng không phải quốc gia riêng lẻ.
- **Kiểm tra trùng lặp** — không có bản ghi trùng lặp.
- **Chuẩn hóa tên quốc gia** — giữ nguyên tiếng Việt, đã nhất quán.

## 3.2 Xử lý giá trị thiếu

Các ô không có dữ liệu được điền bằng 0. Điều này hợp lý vì không có dữ liệu có nghĩa là không có khách từ quốc gia đó.

Sau xử lý: `df_complete` có **2,880 bản ghi** (40 quốc gia × 18 năm × 4 quý).

## 3.3 Trích xuất đặc trưng (Feature Engineering)

- **Biến thời gian:** `quarter_num` (1–4), `time_idx` (năm + quý/4)
- **Đặc trưng trễ:** `lag_1`, `lag_2`, `lag_4` — giá trị quý trước
- **Trung bình trượt:** `rolling_mean_4` — TB 4 quý gần nhất
- **Chia train/test:** Train ≤ 2023, Test ≥ 2024

\\newpage

# 4. Phân tích và khám phá dữ liệu (EDA)

## 4.1 Xu hướng tổng thể

![Hình 1: Lượng khách quốc tế đến Việt Nam theo năm (2008–2026)](output/eda_total_trend.png)

Lượng khách tăng trưởng mạnh từ 3.8 triệu (2009) lên 18.0 triệu (2019) — tăng gần 5 lần. Năm 2020 giảm mạnh xuống 3.7 triệu do COVID-19. Phục hồi từ 2022, đạt 21.2 triệu năm 2025 — vượt mức trước đại dịch.

## 4.2 Top 10 quốc gia nguồn khách

![Hình 2: Top 10 quốc gia nguồn khách hàng đầu](output/eda_top10_countries.png)

- **Trung Quốc** (41.7 triệu) — thị trường lớn nhất
- **Hàn Quốc** (33.0 triệu) — tăng trưởng mạnh nhất
- **Nhật Bản** (10.1 triệu), **Đài Loan** (9.7 triệu), **Hoa Kỳ** (9.1 triệu)

## 4.3 Tính mùa vụ

![Hình 3: Phân tích mùa vụ — Lượng khách trung bình theo quý](output/eda_seasonality.png)

Quý 1 có lượng khách cao nhất (Tết Nguyên Đán, năm mới). Quý 4 có độ biến động lớn nhất.

## 4.4 Tương quan giữa các quốc gia nguồn

![Hình 4: Ma trận tương quan giữa 5 quốc gia nguồn lớn nhất](output/eda_correlation.png)

Trung Quốc–Đài Loan tương quan mạnh (0.89). Hoa Kỳ tương quan thấp với các thị trường châu Á.

## 4.5 Xu hướng theo từng quốc gia

![Hình 5: Xu hướng lượng khách theo từng quốc gia (Top 5)](output/eda_country_trends.png)

Hàn Quốc phục hồi nhanh nhất sau COVID-19. Trung Quốc chịu tác động nặng nề nhất.

\\newpage

# 5. Xây dựng mô hình dự đoán

## 5.1 Linear Regression (Hồi quy tuyến tính)

Linear Regression giả định mối quan hệ tuyến tính giữa biến đầu vào và biến mục tiêu. Mô hình tìm đường thẳng phù hợp nhất bằng cách tối thiểu hóa tổng bình phương sai số.

Đặc trưng đầu vào: năm, số thứ tự quý, chỉ số thời gian, lag_1, lag_4, rolling_mean_4.

## 5.2 Random Forest Regression

Random Forest kết hợp nhiều cây quyết định, mỗi cây huấn luyện trên mẫu bootstrap ngẫu nhiên. Kết quả là trung bình dự đoán của tất cả cây. Giảm overfitting so với cây đơn lẻ.

## 5.3 XGBoost Regressor

XGBoost xây dựng cây quyết định tuần tự, mỗi cây mới sửa lỗi cây trước. Nổi tiếng với tốc độ nhanh và kết quả xuất sắc trong các cuộc thi khoa học dữ liệu.

## 5.4 SARIMA (Mô hình chuỗi thời gian)

SARIMA kết hợp tự hồi quy (AR), lấy sai phân (I), trung bình trượt (MA) và mùa vụ (S). Phù hợp với dữ liệu có tính mùa vụ quarterly. Năm 2021 được loại bỏ khỏi chuỗi.

## 5.5 Chronos-T5 (Foundation Model)

Chronos là mô hình foundation model của Amazon, pretrained trên hàng triệu chuỗi thời gian. Hoạt động theo nguyên lý **zero-shot forecasting** — không cần huấn luyện lại. Cung cấp dự đoán xác suất với khoảng tin cậy.

Thử nghiệm 4 kích thước: Tiny (8M), Small (46M), Base (200M), Large (710M). Chạy trên CPU.

## 5.6 So sánh hiệu suất các mô hình

![Hình 6: So sánh hiệu suất các mô hình dự đoán](output/model_comparison.png)

| Mô hình | MAE | RMSE | R² |
|---------|-----|------|-----|
{comp_table}

**Nhận xét:**

- **Chronos-T5-Base** cho MAE thấp nhất (0.92M) — thấp hơn tất cả mô hình truyền thống.
- **Linear Regression** cho R² tốt nhất (0.48) — nắm bắt xu hướng tuyến tính.
- **SARIMA** cho kết quả kém nhất do khoảng trống dữ liệu 2021 phá vỡ tính liên tục.
- **Random Forest và XGBoost** không vượt trội hơn LR với dữ liệu nhỏ (55 quý).

**Tại sao Base tốt hơn Large?** Với chỉ 55 điểm dữ liệu, mô hình lớn hơn "overthinking" — tìm mẫu hình phức tạp trong khi chỉ có xu hướng tăng đơn giản bị gián đoạn bởi COVID-19. Prior của model lớn (từ tài chính, thời tiết...) không phù hợp với du lịch Việt Nam.

\\newpage

# 6. Dự đoán vs Thực tế (2024–2026)

![Hình 6b: Dự đoán vs Thực tế — Tập test](output/pred_vs_actual.png)

| Quý | Thực tế | LR | Sai lệch | XGBoost | Sai lệch | Chronos | Sai lệch |
|-----|---------|-----|----------|---------|----------|---------|----------|
| 2024 Q1 | 4,642,798 | 3,893,818 | -16.1% | 3,738,777 | -19.5% | 3,784,412 | -18.5% |
| 2024 Q3 | 3,873,045 | 4,236,586 | +9.4% | 3,827,625 | -1.2% | 3,965,012 | +2.4% |
| 2025 Q1 | 6,018,708 | 4,594,127 | -23.7% | 4,506,910 | -25.1% | 4,088,150 | -32.1% |
| 2026 Q1 | 6,762,175 | 5,070,457 | -25.0% | 4,506,910 | -33.4% | 4,334,424 | -35.9% |

**Thách thức "hậu COVID-19":** Không mô hình nào — kể cả Chronos foundation model — có thể dự đoán chính xác sự phục hồi hậu đại dịch. Dữ liệu huấn luyện kết thúc 2023 với ~5 triệu, nhưng 2025–2026 bùng nổ lên 6–7 triệu. Đây là structural break mà mô hình dựa vào lịch sử không thể nắm bắt.

\\newpage

# 7. Dự đoán tương lai

![Hình 7: Dự đoán lượng khách 4 quý tiếp theo](output/forecast_plot.png)

Dự đoán SARIMA cho 4 quý tiếp theo:

| Quý | Dự đoán | Khoảng tin cậy 95% |
|-----|---------|-------------------|
{fc_table}

**Lưu ý:** Khoảng tin cậy rất rộng, phản ánh mức độ không chắc chắn cao do COVID-19 và dữ liệu hạn chế.

\\newpage

# 8. Tổng kết

## 8.1 Kết quả đạt được

1. Phân tích thành công dữ liệu du lịch 2008–2026,chỉ ra xu hướng tăng trưởng và tác động COVID-19.
2. Xác định Trung Quốc và Hàn Quốc là thị trường nguồn lớn nhất.
3. Phát hiện tính mùa vụ: Q1 cao nhất.
4. Xây dựng 5 mô hình, Chronos-T5-Base cho MAE thấp nhất.
5. Dự đoán 4 quý tiếp theo với khoảng tin cậy.

## 8.2 Hạn chế

1. Dữ liệu hạn chế (55 quý huấn luyện).
2. Khoảng trống 2021 phá vỡ tính liên tục chuỗi thời gian.
3. Thiếu đặc trưng bên ngoài (chính sách visa, tỷ giá...).
4. Dự đoán SARIMA không ổn định.

## 8.3 Hướng phát triển

1. Thu thập dữ liệu theo tháng để tăng mẫu.
2. Bổ sung đặc trưng bên ngoài (visa, tỷ giá, sự kiện du lịch).
3. Thử nghiệm Prophet, LSTM.
4. Dự đoán theo từng quốc gia riêng lẻ.
5. Xây dựng dashboard tương tác.
"""

with open('report/bao-cao.md', 'w') as f:
    f.write(md)
print("✓ report/bao-cao.md")

# Convert to LaTeX
subprocess.run(['pandoc', 'report/bao-cao.md', '-o', 'report/bao-cao.tex',
                '--standalone', '--number-sections', '-V', 'geometry:margin=1in',
                '-V', 'mainfont=DejaVu Sans'], check=True)
print("✓ report/bao-cao.tex")

# Convert to PDF
subprocess.run(['pandoc', 'report/bao-cao.md', '-o', 'report/bao-cao.pdf',
                '--pdf-engine=xelatex', '--standalone', '--number-sections',
                '-V', 'geometry:margin=1in'], check=True)
print("✓ report/bao-cao.pdf")
