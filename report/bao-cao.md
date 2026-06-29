---
title: "Phân tích và Dự báo Lượng Khách Du Lịch Quốc tế Đến Việt Nam Sử dụng Dữ liệu Hàng tháng"
subtitle: "Phân tích Dữ liệu với Python — Báo cáo Cuối Kỳ"
author: "Nguyen Dinh Anh Dung (dng-nguyn)"
date: "Tháng 6, 2026"
---

\newpage

# Mục lục

2. Giới thiệu
3. Tổng quan tài liệu
4. Thu thập và parse dữ liệu
5. Tiền xử lý dữ liệu
6. Phân tích khám phá dữ liệu (EDA)
7. Xây dựng mô hình
8. Tối ưu hóa mô hình và Đặc trưng bên ngoài
9. Dự báo và Đánh giá
10. Kết luận
11. Tài liệu tham khảo

\newpage

# Giới thiệu

## Bối cảnh

Du lịch là một trong những ngành kinh tế quan trọng nhất của Việt Nam, đóng góp khoảng 7\% GDP và hỗ trợ khoảng 5,96 triệu việc làm tính đến năm 2024 trên cả nước trong những năm gần đây [1]. Việt Nam đã chào đón lượng khách kỷ lục 18,0 triệu lượt khách quốc tế vào năm 2019, xếp thứ năm ở khu vực Châu Á - Thái Bình Dương (thứ tư ở Đông Nam Á theo dữ liệu PATA nửa đầu năm 2019) [2]. Đại dịch COVID-19 đã khiến lượng khách quốc tế giảm 78,7\% vào năm 2020 [3], với các biên giới bị đóng cửa hiệu quả từ tháng 4 năm 2020 đến tháng 3 năm 2022 (Nghị quyết số 32/NQ-CP). Ngành du lịch Việt Nam sau đó đã phục hồi mạnh mẽ, đạt mức cao kỷ lục mọi thời đại là 21,2 triệu lượt khách vào năm 2025 [4].

Nghiên cứu này phân tích lượng khách du lịch quốc tế hàng tháng đến Việt Nam từ 32 quốc gia nguồn trong giai đoạn 2008--2026 bằng cách sử dụng dữ liệu do Tổng cục Thống kê (GSO) công bố. Độ phân giải hàng tháng cung cấp số lượng quan sát nhiều gấp khoảng 12 lần so với tổng hợp theo quý, cho phép suy luận thống kê mạnh mẽ hơn và độ phân giải mùa vụ chi tiết hơn.

## Mục tiêu

1. Phân tích xu hướng, tính mùa vụ và cơ cấu quốc gia nguồn của lượng khách quốc tế đến Việt Nam
2. So sánh các mô hình dự báo: Hồi quy tuyến tính (Linear Regression), Random Forest, XGBoost, SARIMAX, mô hình nền tảng Chronos-T5, và mô hình phương trình vi phân ngẫu nhiên CIR\#
3. Đánh giá tác động của các đặc trưng bên ngoài (tỷ giá hối đoái, chính sách thị thực/visa) đến độ chính xác của dự báo
4. Tạo dự báo trước 12 tháng với các khoảng tin cậy
5. Tài liệu hóa các giả định, hạn chế và điều kiện biên của mô hình

## Phạm vi

- **Đối tượng:** Lượng khách du lịch quốc tế hàng tháng đến Việt Nam theo quốc gia nguồn
- **Thời gian:** Tháng 1 năm 2008 -- Tháng 5 năm 2026 (không có dữ liệu cho năm 2021 do đóng cửa biên giới vì COVID-19; chỉ có sẵn dữ liệu từ tháng 1 đến tháng 3 năm 2020)
- **Quốc gia:** 32 quốc gia riêng lẻ (không bao gồm dữ liệu tổng hợp theo khu vực)
- **Đối tượng mục tiêu:** $Y^{32}_t$ = tổng lượng khách từ 32 quốc gia được mô hình hóa. Lưu ý: tổng số chính thức GSO tất cả các thị trường ($Y^{\mathrm{official}}_t$) bao gồm các danh mục còn lại không nằm trong 32 quốc gia, do đó $Y^{\mathrm{official}}_t > Y^{32}_t$ (ví dụ: 18,0 triệu so với 17,5 triệu năm 2019). Tất cả việc điều chỉnh mô hình, xác thực và dự báo sử dụng $Y^{32}_t$.
- **Công cụ:** Python 3, pandas, scikit-learn, XGBoost, statsmodels, Chronos-T5, yfinance

\newpage

# Tổng quan tài liệu

Dự báo nhu cầu du lịch đã được nghiên cứu rộng rãi. Song và Witt [5] cung cấp một nghiên cứu nền tảng về mô hình hóa nhu cầu du lịch bằng kinh tế lượng. Một đánh giá toàn diện của Song và các cộng sự [6], bao gồm 211 bài báo, chỉ ra rằng không có phương pháp đơn lẻ nào vượt trội trong mọi bối cảnh; hiệu suất phụ thuộc vào độ chi tiết của dữ liệu, khoảng thời gian dự báo và các điểm gãy cấu trúc (structural breaks).

Đối với các phương pháp chuỗi thời gian, Hyndman và Athanasopoulos [7] trình bày tài liệu tham khảo tiêu chuẩn cho các mô hình họ ARIMA, bao gồm cả các phần mở rộng mùa vụ (SARIMA). Phương pháp Box--Jenkins [8] vẫn là khung cổ điển để nhận dạng, ước lượng và kiểm tra chẩn đoán các mô hình ARIMA.

Các phương pháp học máy tập hợp đã ngày càng trở nên nổi bật. Random Forest [9] giảm phương sai thông qua việc bagging các cây quyết định đã được loại bỏ tương quan. XGBoost [10] xây dựng các cây quyết định một cách tuần tự bằng gradient boosting và regularization, xuất sắc trong việc nắm bắt các tương quan đặc trưng phi tuyến tính nhưng dễ bị quá khớp (overfitting) trên các bộ dữ liệu nhỏ. Sự đánh đổi giữa bias và variance [11] giải thích lý do tại sao các mô hình đơn giản hơn có thể vượt trội hơn các mô hình phức tạp khi kích thước mẫu bị hạn chế.

Các mô hình nền tảng (foundation models) đại diện cho một sự thay đổi mô hình gần đây. Chronos [12], được phát triển bởi Amazon, huấn luyện trước các mô hình dựa trên kiến trúc Transformer trên hàng triệu chuỗi thời gian để dự báo zero-shot mà không cần huấn luyện riêng cho từng tác vụ.

Các mô hình phương trình vi phân ngẫu nhiên (SDE) cũng đã được áp dụng vào du lịch. Orlando và Bufalo [13, 14] đã đề xuất mô hình CIR\#, mở rộng quy trình Cox--Ingersoll--Ross với các phần dư được lọc bằng ARIMA, đạt MAPE là 1,18\% trên dữ liệu du lịch hàng tháng của Ý (288 quan sát). Sự thành công của mô hình phụ thuộc vào việc dữ liệu thỏa mãn các giả định hồi quy về giá trị trung bình (mean-reversion) và có độ chi tiết thời gian đầy đủ.

Đối với bối cảnh Việt Nam, GSO công bố các thống kê lượng khách quốc tế hàng tháng theo quốc gia nguồn [15]. Hội đồng Du lịch và Lữ hành Thế giới (WTTC) [1] và Cục Du lịch Quốc gia Việt Nam [2] cung cấp thêm dữ liệu kinh tế vĩ mô và chính sách bổ sung.

Các khoảng trống dữ liệu khuyết thiếu trong chuỗi thời gian du lịch là một thách thức nổi tiếng. Little và Rubin [16] cung cấp khung nền tảng cho phân tích thống kê với dữ liệu khuyết thiếu. Trong bối cảnh du lịch, các giá trị khuyết thiếu thường phát sinh do các quốc gia chưa được đưa vào khung báo cáo, điều này khác biệt với hiện tượng khuyết thiếu dữ liệu thực sự [6].

\newpage

# Thu thập và parse dữ liệu

## Nguồn dữ liệu

Dữ liệu bao gồm 12 tệp báo cáo HTML-Excel (`t1.xls` đến `t12.xls`), mỗi tệp cho một tháng dương lịch, được tải xuống từ trang web của GSO (`https://www.gso.gov.vn/`) vào tháng 6 năm 2026. Mỗi tệp chứa lượng khách du lịch quốc tế được tổ chức theo quốc gia nguồn, với các cột cho mỗi năm từ năm 2008 hoặc 2009 đến năm 2026.

## Thách thức khi parse

Các tệp `.xls` thực chất là tài liệu HTML có phần mở rộng `.xls`, không phải là các tệp Excel nhị phân tiêu chuẩn. Cấu trúc HTML đặt ra một số thách thức:

1. **Các nút văn bản nằm ngoài thẻ `<td>`.** Các giá trị số xuất hiện dưới dạng các nút văn bản giữa các phần tử `<td>` trống thay vì nằm bên trong chúng. Các trình parse tiêu chuẩn như `pd.read_html()` trả về tất cả các giá trị NaN. Một trình parse tùy chỉnh dựa trên `lxml` đã được xây dựng.

2. **Phạm vi năm không nhất quán.** Các tháng t7 và t9--t12 bao gồm dữ liệu năm 2008; t1--t6 và t8 bắt đầu từ năm 2009. Dữ liệu cho các năm đầu (2009--2011) bị giới hạn ở 11--13 quốc gia.

3. **Định dạng số kiểu Việt Nam.** Dấu chấm được sử dụng làm dấu phân cách hàng nghìn (ví dụ: `33.379` tương đương với 33.379).

4. **Tổng hợp khu vực.** Các mục như "Các thị trường khác" và "Châu Á" đã được loại bỏ.

## Xác minh kết quả parse

Trình parse đã được xác minh đối với tổng số liệu quý đã biết: Hoa Kỳ Q1 2009 = Tháng 1 + Tháng 2 + Tháng 3 = 33.379 + 39.773 + 31.368 = 104.520, khớp chính xác với số liệu quý đã công bố.

## Tóm tắt bộ dữ liệu đã phân tích

| Chỉ số | Giá trị |
|--------|-------|
| Tổng số bản ghi | 4.692 |
| Số quốc gia | 32 |
| Phạm vi ngày | Tháng 7 năm 2008 -- Tháng 5 năm 2026 |
| Năm khuyết thiếu | 2021 (đóng cửa biên giới) |
| Năm không đầy đủ | 2020 (chỉ có từ tháng 1 đến tháng 3) |

## Phân tích độ bao phủ theo quốc gia

Độ bao phủ của các quốc gia rất không đồng đều trong suốt khoảng thời gian:

| Giai đoạn | Số quốc gia mỗi tháng | Ghi chú |
|--------|-------------------|------|
| 2008 (Tháng 7--Tháng 12) | 30 | Chỉ các tháng t7, t9--t12 |
| 2009--2011 | 11--13 | Bị giới hạn nghiêm trọng |
| 2012--2017 | 29--30 | Bao phủ ổn định |
| 2018--2019 | 31 | Mở rộng nhẹ |
| 2020 (Tháng 1--Tháng 3) | 31 | Chỉ trước COVID-19 |
| 2022--2026 | 29--31 | Phục hồi sau COVID-19 |

Khoảng trống bao phủ này có nghĩa là tổng số liệu tổng hợp cho giai đoạn 2009--2011 thấp hơn thực tế một cách nhân tạo. Tất cả các phân tích xu hướng trong báo cáo này sử dụng giai đoạn từ năm 2012 trở đi, nơi độ bao phủ ổn định ở mức 29--31 quốc gia.

\newpage

# Tiền xử lý dữ liệu

## Làm sạch dữ liệu

- **Loại bỏ các tổng hợp khu vực:** Các mục chứa "thi truong khac" (thị trường khác), "Châu" (tiền tố châu lục) và "Tong so" (Tổng số) đã bị loại bỏ.
- **Loại bỏ hàng "Totals"** (Tổng số) xuất hiện dưới dạng một hàng dữ liệu trong một số tệp.
- **Tên quốc gia** được giữ bằng tiếng Việt để phân tích; các bản dịch tiếng Anh được sử dụng trong báo cáo này.
- **Không tìm thấy bản ghi trùng lặp** nào.

## Xử lý giá trị khuyết thiếu

- **Đóng cửa COVID-19 2020--2021:** Đối với phân tích tổng hợp hàng tháng, giai đoạn tháng 4 năm 2020--tháng 12 năm 2021 được điền giá trị 0 (lượng khách = 0) để duy trì tính liên tục của lịch. Biên giới của Việt Nam đã bị đóng cửa hiệu quả trong giai đoạn này; lượng khách báo cáo là không đáng kể (khoảng 157.000--400.000 tùy thuộc vào phương pháp) [3]. Một biến exogenous nhị phân `covid_closed` (1 cho các tháng này, 0 cho các tháng khác) được thêm vào để báo hiệu điểm gãy cấu trúc cho SARIMAX.
- **Các quốc gia thưa thớt:** Ba Lan (Poland, 29 tháng từ năm 2024), Cộng hòa Séc (Czech Republic, 1 tháng vào năm 2026) và Ấn Độ (India, 67 tháng từ năm 2018) có dữ liệu hạn chế. Những quốc gia này được bao gồm trong tổng số liệu tổng hợp khi có sẵn nhưng không được sử dụng cho phân tích riêng từng quốc gia.
- **Các khoảng trống quốc gia-năm-tháng:** Đối với phân tích tổng hợp, các tháng-quốc gia bị khuyết thiếu được coi là có 0 lượt khách. Điều này là hợp lý vì các quốc gia vắng mặt trong khung báo cáo trong một giai đoạn nhất định có lượng khách không đáng kể [16].

## Chia tập Train/Test

- **Tập huấn luyện (Training set):** Tháng 1 năm 2012 -- Tháng 12 năm 2023 (144 tháng, liên tục). Bao gồm các tháng đóng cửa COVID-19 đã được điền giá trị 0 (tháng 4 năm 2020--tháng 12 năm 2021) với `covid_closed = 1`, cung cấp cho mô hình một lịch hoàn chỉnh và tín hiệu điểm gãy cấu trúc.
- **Tập kiểm tra (Test set):** Tháng 1 năm 2024 -- Tháng 12 năm 2025 (24 tháng). Giai đoạn phục hồi hoàn toàn sau COVID-19.
- **Loại trừ khỏi huấn luyện:** 2008--2011 (độ bao phủ quốc gia hạn chế, chỉ 11--13 quốc gia).
- **Khoảng thời gian dự báo (Forecast horizon):** Tháng 1 -- Tháng 12 năm 2026 (12 tháng tiếp theo).

Việc này tạo ra 144 quan sát huấn luyện và 24 quan sát kiểm tra cho phân tích tổng hợp hàng tháng, so với 51 và 10 trong phân tích hàng quý trước đó. Lịch liên tục đảm bảo rằng các đặc trưng trễ (`lag_1`, `lag_12`) bắc cầu khoảng trống COVID-19 một cách chính xác: `lag_1` cho tháng 1 năm 2022 là tháng 12 năm 2021 (= 0), không phải tháng 12 năm 2019.

## Tạo đặc trưng

| Đặc trưng | Mô tả |
|---------|-------------|
| `year` | Năm dương lịch |
| `month` | Tháng (1--12) |
| `time_idx` | Năm + (tháng$-$1)/12, chỉ số thời gian liên tục |
| `lag_1` | Tổng lượt khách của tháng trước |
| `lag_12` | Cùng tháng của năm trước |
| `rolling_mean_12` | Trung bình trượt 12 tháng qua |
| `covid_closed` | Chỉ báo nhị phân (1 cho tháng 4 năm 2020--tháng 12 năm 2021, 0 cho các tháng khác) |
| `exchange_rate_*` | Tỷ giá giao ngay VND so với 8 loại tiền tệ (tùy chọn, trễ 1 tháng) |
| `visa_*` | Các chỉ số chính sách thị thực (tùy chọn) |

Tỷ giá hối đoái được thu thập từ Yahoo Finance dưới dạng tỷ giá giao ngay cuối tháng [17]. Khi được sử dụng làm đặc trưng, tỷ giá được trễ một tháng để tránh rò rỉ dữ liệu, vì tỷ giá tháng hiện tại không có sẵn tại thời điểm dự báo. Các chỉ báo chính sách visa được mã hóa thủ công từ các nguồn của chính phủ Việt Nam [18, 19]:
- `visa_evisa`: Nhị phân, được đặt thành 1 từ tháng 2 năm 2017 (thí điểm e-visa cho 40 quốc gia)
- `visa_evisa_full`: Nhị phân, được đặt thành 1 từ ngày 15 tháng 8 năm 2023 (e-visa 90 ngày phổ quát)
- `covid_restrict`: Thứ hạng (0--1), mã hóa mức độ nghiêm trọng của các hạn chế đi lại

\newpage

**Sử dụng đặc trưng theo mô hình:**

| Đặc trưng | LR | RF | XGBoost | SARIMAX | Chronos | CIR# |
|---------|----|----|---------|---------|---------|------|
| `lag_1` | yes | yes | yes | implicit | no | no |
| `lag_12` | yes | yes | yes | implicit | no | no |
| `rolling_mean_12` | yes | yes | yes | no | no | no |
| `year`, `month`, `time_idx` | yes | yes | yes | no | no | no |
| `covid_closed` | yes | yes | yes | exog | no | no |
| `exchange_rate_*` | opt | opt | opt | no | no | no |
| `visa_*` | opt | opt | opt | no | no | no |
| log-transform target | no | no | no | yes | no | no |
| 2026 forecast method | recursive | recursive | recursive | AR struct | N/A | MC sim |

*Lưu ý: Chronos chỉ sử dụng các giá trị chuỗi thời gian thô. SARIMAX sử dụng cấu trúc ARIMA cho lag và `covid_closed` như exogenous. Tỷ giá và visa bị loại khỏi dự báo 2026 để tránh rò rỉ dữ liệu.*

# Phân tích khám phá dữ liệu (EDA)

## Xuuyên suốt và bối cảnh phạm vi bao phủ

![Tổng lượt khách hàng năm với biểu đồ chồng số lượng quốc gia. Lưu ý rằng tổng số năm 2009--2011 bị giảm do phạm vi bao phủ hạn chế (11--13 quốc gia).](output/eda_total_trend.png)

Biểu đồ cột hiển thị tổng lượng khách hàng năm (trục trái) với số lượng quốc gia báo cáo được hiển thị chồng lên (trục phải, đường màu đỏ). Các quan sát chính:

- **2009--2011:** Chỉ có 11--13 quốc gia báo cáo, làm cho tổng số liệu thấp hơn thực tế một cách nhân tạo.
- **2012--2019:** Tăng trưởng ổn định với 29--31 quốc gia báo cáo; lượng khách tăng từ 6,8 triệu (2012) lên 18,0 triệu (2019).
- **2020:** Chỉ có dữ liệu từ tháng 1 đến tháng 3; biên giới đóng cửa từ tháng 4.
- **2022--2025:** Phục hồi mạnh mẽ sau COVID-19, đạt 21,2 triệu lượt khách vào năm 2025 [4].

## Các quốc gia nguồn hàng đầu

![Top 10 quốc gia nguồn theo lượng khách tích lũy (2008--2026).](output/eda_top10_countries.png)

Trung Quốc (tích lũy 41,7 triệu) và Hàn Quốc (33,0 triệu) là các thị trường nguồn thống trị. Tổng số này bao gồm tất cả các năm có sẵn bao gồm cả giai đoạn bị giới hạn phạm vi bao phủ 2009--2011.

## Tỷ lệ tăng trưởng (2012--2019)

![Tốc độ tăng trưởng hàng năm trung bình theo quốc gia (2012--2019).](output/eda_growth_rate.png)

Tỷ lệ tăng trưởng phản ánh **các thị trường nguồn mới nổi** (Hồng Kông, Tây Ban Nha, Ý, Philippines) chứ không phải là các thị trường khối lượng lớn đã ổn định. Trung Quốc và Hàn Quốc, hai nguồn lớn nhất, thể hiện mức tăng trưởng vừa phải từ một nền tảng cao.

## Tính mùa vụ theo tháng

![Mẫu mùa vụ hàng tháng trên tất cả các năm và quốc gia (2012--2019, 2022--2025).](output/eda_seasonality.png)

- **Tháng 1--Tháng 2** có lượng khách cao nhất, được thúc đẩy bởi kỳ nghỉ Tết Nguyên đán và du lịch mùa đông.
- **Tháng 6--Tháng 8** cho thấy một đỉnh phụ (du lịch mùa hè).
- **Tháng 11--Tháng 12** có lượng khách trung bình thấp nhất.

## Tương quan giữa các thị trường nguồn

![Ma trận tương quan của 5 quốc gia nguồn hàng đầu (lượng khách hàng tháng).](output/eda_correlation.png)

Trung Quốc và Đài Loan thể hiện mức tương quan cao nhất (khoảng 0,89), phản ánh vị trí địa lý chung và các mẫu du lịch tương tự. Hoa Kỳ cho thấy mức tương quan thấp hơn với các thị trường châu Á, cho thấy động lực nhu cầu độc lập.

## Xu hướng theo từng quốc gia

![Lượng khách hàng tháng của 5 quốc gia nguồn hàng đầu (2008--2026).](output/eda_country_trends.png)

- **Hàn Quốc** phục hồi nhanh nhất sau COVID-19.
- **Trung Quốc** hứng chịu đợt sụt giảm sâu nhất và phục hồi chậm nhất.
- **Nhật Bản** cho thấy mức tăng trưởng ổn định nhưng vừa phải trong suốt giai đoạn.

\newpage

# Xây dựng mô hình

Phần này mô tả từng mô hình dự báo được áp dụng cho **tổng lượng khách hàng tháng** (tổng hợp trên tất cả các quốc gia). Mỗi mô hình sử dụng bộ đặc trưng được mô tả trong Phần 4.3. Các mô hình được đánh giá trên tập kiểm tra (Tháng 1 năm 2024 -- Tháng 12 năm 2025, 24 tháng) bằng bốn chỉ số:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R$^2$** (Hệ số xác định): Tỷ lệ phương sai được giải thích, tính theo $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{\mathrm{test}})^2}$. Giá trị âm cho thấy hiệu suất kém hơn so với việc dự đoán giá trị trung bình của tập kiểm tra, xảy ra khi các mô hình được huấn luyện trên dữ liệu 2012--2023 không thể ngoại suy xu hướng tăng trưởng sau COVID-19 [11].

## Hồi quy tuyến tính

Hồi quy tuyến tính giả định một mối quan hệ tuyến tính giữa các đặc trưng $\mathbf{x}$ và biến mục tiêu $y$:

$$y = \mathbf{w}^T \mathbf{x} + b$$

Mô hình tối thiểu hóa tổng bình phương phần dư (Ordinary Least Squares):

$$\min_{\mathbf{w}, b} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

**Các giả định:** Tính tuyến tính, tính độc lập của các phần dư, tính đồng nhất phương sai (homoscedasticity), các sai số phân phối chuẩn. Với 144 mẫu huấn luyện và 6 đặc trưng, tỷ lệ quan sát trên tham số (24:1) là đầy đủ.

**Khả năng áp dụng:** Đặc trưng `time_idx` cung cấp một tín hiệu tuyến tính mạnh mẽ cho xu hướng đi lên tổng thể. Với 144 mẫu huấn luyện, sự tinh giản của mô hình là một lợi thế so với các giải pháp thay thế phức tạp [11].

**Hạn chế:** Không thể nắm bắt được các mẫu phi tuyến tính, các chu kỳ mùa vụ nằm ngoài những gì các đặc trưng mã hóa theo tháng cung cấp, hoặc các điểm gãy cấu trúc.

## Hồi quy Random Forest

Random Forest [9] là một tập hợp gồm $T$ cây quyết định, mỗi cây được huấn luyện trên một mẫu bootstrap với việc lấy mẫu đặc trưng ngẫu nhiên:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

Việc trung bình hóa các cây quyết định đã được loại bỏ tương quan giúp giảm phương sai trong khi vẫn duy trì khả năng mô hình hóa phi tuyến tính (bagging).

**Các giả định:** Các mẫu độc lập và phân phối chuẩn (i.i.d.). Không yêu cầu tính dừng (stationarity) hoặc tính tuyến tính.
**Các siêu tham số chính:** `n_estimators` = 200, `max_depth` = 10 (để giới hạn quá khớp trên 144 mẫu), `min_samples_split` = 2.

**Lợi thế so với phân tích hàng quý:** Với 144 mẫu huấn luyện (so với 51 mẫu theo quý), Random Forest có thể xây dựng các cây quyết định đa dạng hơn và có khả năng tổng quát hóa tốt hơn.

## Mô hình XGBoost Regressor

XGBoost [10] xây dựng các cây quyết định **một cách tuần tự**, với mỗi cây sửa chữa sai số của tập hợp cây trước đó:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

Tại mỗi bước, một cây mới sẽ tối thiểu hóa một hàm mục tiêu có regularization:

$$\mathcal{L}^{(k)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)) + \Omega(f_k)$$

trong đó $\ell$ là sai số bình phương và $\Omega$ phạt độ phức tạp của cây.

**Các giả định:** Các mẫu i.i.d. Regularization là rất quan trọng để ngăn chặn việc ghi nhớ nhiễu huấn luyện [10].

**Các siêu tham số chính:** `n_estimators` = 200, `max_depth` = 6, `learning_rate` = 0.1.

## SARIMAX$(1,1,1)(1,1,1)_{12}$

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) [7, 8] là một mô hình thống kê cổ điển phân tích một chuỗi thời gian thành các thành phần tự hồi quy (autoregressive), sai phân (differencing), trung bình trượt (moving-average) và mùa vụ, được bổ sung thêm các biến hồi quy exogenous. Để ổn định phương sai và đảm bảo biến đổi ngược không âm, mô hình được điều chỉnh trên $z_t = \log(y_t + 1)$:

$$\Phi_P(B^s)\,\phi_p(B)\,(1-B^s)^D\,(1-B)^d\, z_t = \sum_{j} \beta_j x_{j,t} + \Theta_Q(B^s)\,\theta_q(B)\,\varepsilon_t$$

trong đó $B$ là toán tử backshift ($B y_t = y_{t-1}$), $s = 12$ là chu kỳ mùa vụ, $x_{j,t}$ là các biến exogenous, và $\varepsilon_t$ là nhiễu trắng.

where $z_t = \log(y_t + 1)$ is the log-transformed target variable.

Mô hình được sử dụng ở đây, SARIMAX$(1,1,1)(1,1,1)_{12}$, áp dụng sai phân bậc một ở cả cấp độ thông thường và cấp độ mùa vụ, với một số hạng tự hồi quy và một số hạng trung bình trượt ở mỗi cấp độ. Một biến exogenous nhị phân `covid_closed` (1 cho tháng 4 năm 2020--tháng 12 năm 2021, 0 cho các tháng khác) báo hiệu điểm gãy cấu trúc do đóng cửa biên giới, cho phép mô hình học được sự thay đổi mức mà không bị gián đoạn trong lịch.

**Các giả định:** Tính dừng sau khi lấy sai phân; các phần dư là nhiễu trắng. Giai đoạn đóng cửa COVID-19 (tháng 4 năm 2020--tháng 12 năm 2021) được điền giá trị 0 với `covid_closed = 1`, duy trì tính liên tục của lịch để các đặc trưng trễ bắc cầu khoảng trống một cách chính xác.

## Chronos-T5

Chronos [12] là một họ các mô hình nền tảng dựa trên Transformer được huấn luyện trước cho chuỗi thời gian. Không giống như các mô hình khác học từ 144 mẫu huấn luyện, Chronos không được huấn luyện trên dữ liệu này; nó đã được huấn luyện trước trên hàng triệu chuỗi thời gian từ nhiều lĩnh vực khác nhau. Nó hoạt động bằng cách mã hóa (tokenizing) các giá trị chuỗi thời gian và tạo ra các dự báo xác suất thông qua lấy mẫu tự hồi quy.

| Mô hình | Tham số | Cửa sổ ngữ cảnh |
|-------|-----------|---------------|
| chronos-t5-tiny | 8M | 512 tokens |
| chronos-t5-small | 46M | 512 tokens |
| chronos-t5-base | 200M | 512 tokens |

**Lợi thế của dữ liệu hàng tháng:** Với 144 điểm ngữ cảnh (so với 55 điểm theo quý), mô hình có tín hiệu dày đặc hơn để nhận dạng mẫu. Cửa sổ ngữ cảnh 512 token hoàn toàn nằm trong giới hạn cho phép.

## Mô hình phương trình vi phân ngẫu nhiên CIR\#

Mô hình CIR\# [13, 14] mở rộng phương trình vi phân ngẫu nhiên (SDE) Cox--Ingersoll--Ross để dự báo du lịch với dữ liệu bị gián đoạn:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

trong đó $r(t)$ là lượng khách du lịch tổng hợp hàng tháng (người/tháng), $\kappa > 0$ là tốc độ hồi quy về trung bình (mỗi tháng), $\theta$ là mức trung bình dài hạn (người/tháng), $\sigma$ là tham số độ biến động (người/tháng$^{1/2}$), và $dW(t)$ là bước tăng của chuyển động Brown chuẩn. Các tham số được ước lượng qua OLS trên các chênh lệch rời rạc $\Delta r = r_{t+1} - r_t$ đối với $r_t$, cho $\hat{\kappa} = -\hat{\beta}$ và $\hat{\theta} = \hat{\alpha}/\hat{\kappa}$.

Orlando và Bufalo [13] báo cáo MAPE là 1,18\% trên dữ liệu du lịch hàng tháng của Ý (288 quan sát), giảm 70\% sai số so với SARIMA và Holt--Winters. Phần mở rộng CIR\# thay thế chuyển động Brown bằng các phần dư được lọc bằng ARIMA và phân chia dữ liệu thành các mẫu phụ xung quanh các điểm gãy cấu trúc.

**Đánh giá trên dữ liệu hàng tháng của Việt Nam:** Với 144 quan sát huấn luyện, CIR\# có đủ dữ liệu để ước lượng tham số MLE nhưng phải đối mặt với ba thách thức cơ bản: (1) ngành du lịch của Việt Nam thể hiện một xu hướng đi lên mạnh mẽ vi phạm giả định hồi quy về giá trị trung bình (số hạng $\kappa(\theta - r(t))$); (2) mức giảm 80\% do COVID-19 sau đó là sự phục hồi hoàn toàn tạo ra biến động log-return cực lớn được khuếch đại bởi số hạng khuếch tán $\sqrt{r}$; (3) mô hình được thiết kế cho các quy trình dừng, có xu hướng hồi quy về giá trị trung bình [14].

## So sánh mô hình

![So sánh hiệu suất mô hình (MAE, MAPE, R$^2$).](output/model_comparison.png)

| Mô hình            | MAE       | RMSE      | MAPE   | R²       |
|--------------------|-----------|-----------|--------|----------|
| Chronos-T5-small   | 170.625   | 214.069   | 10,77% | −0,03    |
| Linear Regression  | 275.526   | 340.495   | 19,79% | 0,24     |
| Random Forest      | 290.287   | 334.896   | 19,80% | 0,27     |
| XGBoost            | 274.730   | 320.581   | 17,75% | 0,11     |
| SARIMAX (log)        | 314.479   | 355.003   | 20,39% | −0,71    |
| CIR#               | 396.833   | 467.793   | 25,12% | −1,14    |

*Lưu ý: Các chỉ số chính xác được điền từ đầu ra của mô hình (model_results.csv).*

**Các quan sát chính:**

- **Chronos-T5-small đạt MAPE tốt nhất** (10,77\%) với tư cách là mô hình nền tảng zero-shot, thể hiện khả năng tổng quát hóa mạnh mẽ mà không cần huấn luyện trên bộ dữ liệu này.
- **Linear Regression, Random Forest và XGBoost** có hiệu suất tương tự nhau (MAPE $\approx$ 19,8\%), trong đó Random Forest đạt R$^2$ cao nhất (0,27). Các mô hình này hưởng lợi từ đặc trưng `lag_1` nhưng bị giới hạn bởi điểm gãy cấu trúc giữa tập huấn luyện (2012--2019 + 2022--2023) và tập kiểm tra (2024--2025).
- Dữ liệu hàng tháng cung cấp 144 mẫu huấn luyện (so với 51 mẫu theo quý), cho phép các mô hình dựa trên cây quyết định tổng quát hóa tốt hơn.
- SARIMAX với mục tiêu biến đổi log và biến exogenous `covid_closed` (MAPE = 26,87\%, R$^2$ = $-$0,71). Biến đổi log $\log(y+1)$ đảm bảo các khoảng tin cậy không âm một cách toán học mà không cần cắt thủ công, nhưng R$^2$ trên tập kiểm tra cho thấy mô hình vẫn gặp khó khăn trong việc ngoại suy quỹ đạo tăng trưởng sau COVID-19 từ khuôn khổ quy trình dừng.
- CIR\# tiếp tục thất bại (MAPE = 28,54\%, R$^2$ = $-$1,14) do vi phạm giả định hồi quy về giá trị trung bình, mặc dù có dữ liệu hàng tháng như các tác giả gốc đã khuyến nghị [13].
- Ba mô hình (Linear Regression, Random Forest, XGBoost) đạt giá trị R$^2$ dương, trong khi SARIMAX, Chronos và CIR\# vẫn ở mức âm do điểm gãy cấu trúc sau COVID-19.
- Sự thống trị của `lag_1` (69,8\%) có nghĩa là các mô hình dựa trên cây hoạt động chủ yếu như các mô hình dự báo naive một bước. Điều này hoạt động tốt trên tập kiểm tra 24 tháng (nơi giá trị lag thực tế của mỗi tháng có sẵn) nhưng sẽ suy giảm đáng kể trong các kịch bản dự báo đa bước khi các giá trị lag phải được dự đoán đệ quy.

**Lưu ý quan trọng: dự báo một bước vs đa bước.** Bảng so sánh mô hình trên đánh giá độ chính xác *một bước trước*: mỗi tháng kiểm tra sử dụng giá trị lag thực tế của tháng trước. Dự báo 2026, tuy nhiên, yêu cầu dự báo *đa bước đệ quy*: dự báo của mỗi tháng được đưa trở lại làm đầu vào `lag_1` cho tháng tiếp theo, tích lũy sai số. Các chỉ số một bước được báo cáo ở đây do đó đại diện cho giới hạn trên của hiệu suất dự báo. Đánh giá đa bước thích hợp (rolling-origin hoặc blocked time-series) sẽ cần thiết để đánh giá độ chính xác dự báo 12 tháng.
# Tối ưu hóa mô hình và Đặc trưng bên ngoài

## Tối ưu hóa siêu tham số

**Random Forest** đã được tối ưu hóa thông qua GridSearchCV (TimeSeriesSplit (n_splits=3)):

- `n_estimators`: {100, 200, 300}
- `max_depth`: {5, 10, 15, None}
- `min_samples_split`: {2, 5, 10}

**XGBoost** đã được tối ưu hóa thông qua RandomizedSearchCV (50 vòng lặp):

- `n_estimators`: 100--500
- `max_depth`: 3--9
- `learning_rate`: 0,01--0,2
- `subsample`: 0,7--1,0
- `colsample_bytree`: 0,7--1,0

## Đặc trưng bên ngoài

Tỷ giá hối đoái (VND so với KRW, CNY, USD, JPY, TWD, MYR, THB, RUB) được lấy từ Yahoo Finance [17] dưới dạng tỷ giá giao ngay cuối tháng. Các chỉ số chính sách thị thực được mã hóa từ các nguồn của chính phủ Việt Nam [18, 19].

**Kết quả:** Các đặc trưng bên ngoài mang lại sự cải thiện không đáng kể cho Hồi quy tuyến tính nhưng làm giảm hiệu suất của các mô hình dựa trên cây quyết định do quá khớp. Phân tích tầm quan trọng của đặc trưng (feature importance) cho thấy `lag_1` và `rolling_mean_12` vẫn là các đặc trưng thống trị. Điểm gãy cấu trúc sau COVID-19 vẫn là thách thức cơ bản mà chỉ riêng các đặc trưng bên ngoài không thể giải quyết được.

\newpage

# Dự báo và Đánh giá

## Dự đoán so với Thực tế (Tập kiểm tra)

![Lượng khách hàng tháng dự đoán so với thực tế cho tập kiểm tra (2022--2025).](output/pred_vs_actual.png)

XGBoost và Random Forest theo dõi giá trị thực tế chặt chẽ nhất (MAPE $\approx$ 20\%), trong khi SARIMAX và CIR\# phân kỳ đáng kể. Chronos-T5-small cung cấp dự báo zero-shot cạnh tranh tốt nhất (MAPE = 10,77\%). Các mô hình dựa trên cây (RF, XGBoost) cho thấy sự biến động nhiều hơn trong các dự đoán, trong khi Hồi quy tuyến tính tạo ra các dự báo mượt mà hơn.

**Ghi chú phương pháp dự báo.** Tất cả bốn mô hình tạo dự báo trước 12 tháng cho năm 2026. Các mô hình dựa trên cây (Hồi quy tuyến tính, Random Forest, XGBoost) sử dụng chiến lược đệ quy: dự báo của mỗi tháng được đưa trở lại làm đặc trưng `lag_1` cho tháng tiếp theo. Cách tiếp cận này tích lũy sai số tại mỗi bước nhưng nắm bắt được các mẫu phi tuyến mà các mô hình cây đã học. SARIMAX sử dụng cấu trúc tự hồi quy để tạo dự báo đa bước trực tiếp. Giá trị trung bình tập hợp (hiển thị bằng màu đỏ) lấy trung bình của cả bốn mô hình, cung cấp dự báo mạnh mẽ hơn bất kỳ mô hình đơn lẻ nào. Dải bóng hiển thị phạm vi giữa mô hình lạc quan nhất và bi quan nhất — thước đo dễ hiểu hơn khoảng tin cậy SARIMAX. Hơn nữa, các đặc trưng bên ngoài động (như tỷ giá hối đoái) đã bị loại khỏi các mô hình dự báo ngoài mẫu cuối cùng năm 2026 để tránh rò rỉ dữ liệu, nghĩa là các dự báo đa bước dựa hoàn toàn vào các mẫu tự hồi quy, chỉ số lịch và chỉ báo chính sách.

## Dự báo SARIMAX 12 tháng (2026)

![Dự báo 12 tháng của SARIMAX cho năm 2026 với khoảng tin cậy 95\%.](output/forecast_plot.png)

| Tháng | Dự báo | KTC 95\% Dưới | KTC 95\% Trên |
|-------|--------|----------------|----------------|
| Tháng 1/2026 | 1.538.229 | 940.658 | 2.515.418 |
| Tháng 2/2026 | 1.334.457 | 700.080 | 2.543.676 |
| Tháng 3/2026 | 1.077.130 | 491.657 | 2.359.791 |
| Tháng 4/2026 | 1.164.874 | 475.096 | 2.856.116 |
| Tháng 5/2026 | 1.074.181 | 395.703 | 2.915.977 |
| Tháng 6/2026 | 1.400.343 | 470.662 | 4.166.384 |
| Tháng 7/2026 | 1.546.696 | 477.592 | 5.009.008 |
| Tháng 8/2026 | 1.713.283 | 488.855 | 6.004.510 |
| Tháng 9/2026 | 1.586.391 | 420.225 | 5.988.773 |
| Tháng 10/2026 | 1.601.608 | 395.427 | 6.487.018 |
| Tháng 11/2026 | 1.436.842 | 331.758 | 6.222.945 |
| Tháng 12/2026 | 1.854.630 | 401.648 | 8.563.820 |

Để ngăn chặn dự báo các lượt khách đến âm không thể có về mặt vật lý và ổn định phương sai, biến mục tiêu đã được biến đổi log ($\log(y+1)$) trước khi điều chỉnh SARIMAX. Các dự báo và khoảng tin cậy kết quả đã được chuyển đổi ngược lại về thang đo gốc bằng $\exp(\cdot) - 1$, tự nhiên giới hạn các cận dưới khoảng tin cậy ở mức 0 mà không cần cắt thủ công tùy ý. Cách tiếp cận này tạo ra các khoảng tin cậy rộng hơn so với mô hình thang đo gốc, phản ánh sự không đối xứng vốn có trong các quy trình nhân.

## Dự báo theo quốc gia nguồn (2026)


**Lưu ý về biến đổi ngược:** Biến đổi ngược $\exp(\hat{z}_t) - 1$ tạo ra một dự báo theo thang đo trung vị trên thang đo gốc, không phải là giá trị trung bình có điều kiện. Dự báo trung bình không lệch sẽ cần một sửa đổi lognorm $\exp(\hat{z}_t + \hat{\sigma}^2_t/2) - 1$ hoặc xử lý dựa trên mô phỏng. Khoảng tin cậy là không đối xứng trên thang đo gốc, phản ánh bản chất nhân của quá trình log-biến đổi.

Các mô hình tập hợp (trung bình của Hồi quy tuyến tính, Random Forest, XGBoost và SARIMAX) được điều chỉnh cho 5 quốc gia nguồn hàng đầu. Các mô hình cây sử dụng dự báo đệ quy để tạo dự đoán trước 12 tháng; SARIMAX sử dụng cấu trúc tự hồi quy trực tiếp.

| Tháng | Hàn Quốc | Trung Quốc | Campuchia | Nhật Bản | Nga |
|-------|----------|------------|-----------|----------|-----|
| Tháng 1/2026 | 389.721 | 212.894 | 32.843 | 60.466 | 14.745 |
| Tháng 2/2026 | 407.507 | 230.786 | 31.082 | 60.571 | 15.503 |
| Tháng 3/2026 | 363.183 | 226.083 | 31.519 | 62.357 | 14.771 |
| Tháng 4/2026 | 350.430 | 227.242 | 30.196 | 59.207 | 13.652 |
| Tháng 5/2026 | 354.600 | 245.202 | 29.337 | 60.160 | 13.218 |
| Tháng 6/2026 | 363.588 | 252.661 | 28.884 | 58.926 | 12.304 |
| Tháng 7/2026 | 371.555 | 265.062 | 30.280 | 60.739 | 11.455 |
| Tháng 8/2026 | 420.453 | 277.285 | 29.165 | 75.738 | 11.884 |
| Tháng 9/2026 | 400.238 | 269.141 | 32.144 | 78.329 | 12.094 |
| Tháng 10/2026 | 406.438 | 271.010 | 32.529 | 64.637 | 12.410 |
| Tháng 11/2026 | 411.074 | 288.898 | 34.186 | 66.296 | 14.932 |
| Tháng 12/2026 | 429.562 | 309.729 | 34.750 | 70.569 | 15.117 |
| **Tổng** | **4.668.349** | **3.075.993** | **376.915** | **777.995** | **162.085** |

*Bảng hiển thị giá trị trung bình tập hợp của Hồi quy tuyến tính, Random Forest, XGBoost và SARIMAX. Phạm vi bất đồng mô hình: 8,2M–10,1M tổng.*

5 quốc gia hàng đầu chiếm 52,3\% tổng dự báo tập hợp năm 2026 (9,1 trên 17,3 triệu). Hàn Quốc được dự báo vẫn là thị trường nguồn lớn nhất, tiếp theo là Trung Quốc. Dải bóng trong hình hiển thị phạm vi giữa dự báo cao nhất và thấp nhất của các mô hình — thước đo dễ hiểu hơn khoảng tin cậy SARIMAX.


\newpage

## Kiểm chứng dự báo (Tháng 1–5/2026)

Với dữ liệu thực tế có sẵn cho 5 tháng đầu năm 2026, chúng ta có thể đánh giá dự báo chỉ-SARIMAX so với thực tế tổng hợp 32 quốc gia ($Y^{32}_t$) (dự báo tập hợp sử dụng thành phần SARIMAX tương tự cho dự báo tổng hợp).

![Dự báo so với thực tế cho tháng 1—5/2026 (tổng hợp và các quốc gia nguồn hàng đầu).](output/forecast_validation.png)

| Tháng | Thực tế | Dự báo | Sai số |
|-------|---------|--------|--------|
| Tháng 1/2026 | 1.641.403 | 1.169.591 | −28,7% |
| Tháng 2/2026 | 2.124.123 | 1.225.010 | −42,3% |
| Tháng 3/2026 | 1.540.586 | 1.077.132 | −30,1% |
| Tháng 4/2026 | 1.601.269 | 1.164.874 | −27,3% |
| Tháng 5/2026 | 1.553.853 | 1.074.181 | −30,8% |
| **MAPE** | | | **31,9%** |

**Kiểm chứng theo quốc gia:**

| Quốc gia | MAPE | Ghi chú |
|----------|------|---------|
| Hàn Quốc | 8,3% | Phù hợp tốt nhất; mô hình nắm bắt mẫu mùa vụ tốt |
| Trung Quốc | 48,1% | Đánh giá thấp nhất quán; tăng trưởng cấu trúc từ 2024 không được nắm bắt |
| Nhật Bản | 25,9% | Ngoại lệ tháng 2/2026 (844K so với ~67K thông thường) có thể là bất thường dữ liệu nguồn |
| Campuchia | 50,3% | Thay đổi chế độ biên giới: miễn thị thực + đường bay mới + du lịch y tế; tháng 1/2026 đạt 223K (gấp 3 lần tháng 12/2025) |

MAPE tổng hợp là 31,9% xác nhận rằng mô hình SARIMAX đánh giá thấp nhất quán sự tăng tốc tăng trưởng sau COVID-19. Hàn Quốc là quốc gia duy nhất được dự báo tốt (MAPE = 8,3%), vì quỹ đạo tăng trưởng của nó gần nhất với phân phối huấn luyện 2012–2023. Trường hợp Campuchia minh họa giới hạn cơ bản: mô hình không thể dự đoán sự thay đổi chế độ gấp 10 lần.


# Kết luận

## Kết quả chính

1. **Dữ liệu hàng tháng cải thiện đáng kể phân tích.** Với 144 quan sát huấn luyện (so với 51 quan sát theo quý), các mô hình học máy có lượng dữ liệu gần gấp ba để học các mẫu dữ liệu. SARIMAX với $s = 12$ giải quyết cấu trúc mùa vụ chi tiết hơn (kỳ nghỉ Tết, đỉnh mùa hè) mà $s = 4$ không thể nắm bắt.

2. **Sai lệch bao phủ vẫn tồn tại.** Chỉ có 11--13 quốc gia báo cáo dữ liệu hàng tháng trong giai đoạn 2009--2011, so với 29--31 từ năm 2012 trở đi. Các phân tích xu hướng tổng hợp nên sử dụng năm 2012 làm thời điểm bắt đầu.

3. **Trung Quốc và Hàn Quốc thống trị** thị trường du lịch, chiếm lượng khách tích lũy lớn nhất. Các thị trường mới nổi (Hồng Kông, Tây Ban Nha, Ý, Philippines) tăng trưởng nhanh nhất trong giai đoạn 2012--2019.

4. **Tính mùa vụ mạnh mẽ:** Tháng 1--Tháng 2 liên tục có lượng khách cao nhất (Tết Nguyên đán + du lịch mùa đông), với đỉnh mùa hè phụ vào tháng 6--Tháng 8.

5. **Điểm gãy cấu trúc sau COVID-19** vẫn là thách thức thống trị. Ba mô hình (Random Forest, Linear Regression, XGBoost) đạt giá trị R$^2$ dương (0,11--0,27), trong khi Chronos, SARIMAX và CIR\# vẫn ở mức âm ($-$0,03, $-$0,71, $-$1,14) vì phân phối của tập kiểm tra (phục hồi 2024--2025) khác biệt cơ bản so với phân phối của tập huấn luyện (2012--2023).

6. **CIR\# thất bại trên dữ liệu có xu hướng.** Mặc dù có dữ liệu hàng tháng như khuyến nghị của Orlando và Bufalo [13], giả định hồi quy về giá trị trung bình của mô hình bị vi phạm bởi quỹ đạo du lịch có xu hướng đi lên của Việt Nam. Kết quả này phù hợp với điều kiện biên được tài liệu hóa cho mô hình [13, 14].

7. **Các đặc trưng bên ngoài mang lại sự cải thiện không đáng kể** cho Hồi quy tuyến tính nhưng không có tác dụng đối với các mô hình dựa trên cây quyết định, vốn bị quá khớp trên các chiều đặc trưng bổ sung.

## Hạn chế

1. **Khoảng trống COVID-19 (2020--2021):** Điền giá trị 0 cho giai đoạn đóng cửa COVID-19 duy trì tính liên tục của lịch; biến exogenous `covid_closed` báo hiệu điểm gãy cấu trúc cho SARIMAX. Tuy nhiên, các tháng điền 0 vẫn đại diện cho một sự khác biệt so với quy trình tạo dữ liệu thực sự.
2. **Tập huấn luyện hạn chế (144 tháng):** Mặc dù tốt hơn đáng kể so với 51 điểm hàng quý, con số này vẫn hạn chế độ phức tạp của mô hình so với 288 quan sát hàng tháng được sử dụng trong nghiên cứu CIR\# của Ý [13].
3. **Thiếu các đặc trưng bên ngoài:** GDP của quốc gia nguồn, công suất chuyến bay, lượng tìm kiếm trên Google Trends và giá dầu không được bao gồm.
4. **Sự không nhất quán về độ bao phủ (2009--2011):** Chỉ có 11--13 quốc gia mỗi tháng làm giới hạn độ tin cậy của các số liệu thống kê tổng hợp cho những năm đầu.
5. **Chưa thực hiện các chẩn đoán chính thức:** Các thử nghiệm tính chuẩn của phần dư, thử nghiệm tính dừng (ADF, KPSS) và thử nghiệm phương sai sai số thay đổi (heteroscedasticity) sẽ giúp củng cố phân tích.

## Hướng phát triển tương lai

1. **GDP quốc gia nguồn và công suất chuyến bay** làm các chỉ báo dẫn dắt
2. **Lượng tìm kiếm trên Google Trends** cho các truy vấn về điểm đến
3. **Các mô hình regime-switching** xử lý rõ ràng các chuyển đổi trước COVID-19, trong COVID-19 và sau COVID-19
4. **Các mô hình riêng cho từng quốc gia** (các quốc gia khác nhau thể hiện các quỹ đạo phục hồi khác nhau)
5. **Các phương pháp tập hợp** kết hợp Hồi quy tuyến tính (cho xu hướng) với Chronos (cho nhận dạng mẫu)
6. **Chẩn đoán phần dư chính thức** và kiểm tra tính dừng

\newpage

# Tài liệu tham khảo

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

[13] G. Orlando và M. Bufalo, "Improved tourism demand forecasting with CIR\# model: the case of disrupted data patterns in Italy," *Tourism Review*, vol. 79, no. 2, pp. 445--470, 2023, doi: 10.1108/TR-04-2023-0194.

[14] G. Orlando và M. Bufalo, "The CIR\# model for time series forecasting," *Technological and Economic Development of Economy*, vol. 29, no. 5, pp. 1403--1427, 2023.

[15] General Statistics Office of Vietnam, "Quarterly and monthly international arrival statistics by country," GSO, Hanoi, 2008--tháng 5 năm 2026 (dữ liệu một phần). [Online]. Available: https://www.gso.gov.vn/

[16] R. J. A. Little and D. B. Rubin, *Statistical Analysis with Missing Data*, 2nd ed. Wiley, 2002.

[17] Yahoo Finance, "Historical exchange rates," Yahoo, 2025. [Online]. Available: https://finance.yahoo.com/

[18] Vietnam Immigration Department, "Pilot e-visa system for foreign visitors," effective Feb. 1, 2017, initially for 40 countries.

[19] Vietnam National Assembly, "Resolution on extension and amendment of e-visa policy," approved Jun. 24, 2023, effective Aug. 15, 2023.


[20] VietnamPlus, "Russian tourists to Vietnam surge in 2024," Vietnam News Agency, 2024. [Online]. Available: https://en.vietnamplus.vn/

[21] VnExpress International, "Cambodia thay thế Đài Loan trở thành nguồn khách du lịch lớn thứ 3 của Việt Nam," tháng 2/2026.

[22] Scikit-learn developers, "Scikit-learn: Machine Learning in Python," 2025. [Online]. Available: https://scikit-learn.org/

[23] T. Chen et al., "XGBoost Documentation," 2025. [Online]. Available: https://xgboost.readthedocs.io/

[24] Statsmodels developers, "Statsmodels: Statistical Modeling and Econometrics in Python," 2025. [Online]. Available: https://www.statsmodels.org/

[25] Amazon Science, "Chronos Forecasting," GitHub, 2024.

[26] Pacific Asia Travel Association, "Visitor Arrivals to Asia Pacific Destinations," PATA, nửa đầu năm 2019. [Online]. Available: https://github.com/amazon-science/chronos-forecasting
