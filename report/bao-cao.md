---
title: "Phân tích và Dự báo Lượng Khách Du Lịch Quốc tế Đến Việt Nam Sử dụng Dữ liệu Hàng tháng"
subtitle: "Phân tích Dữ liệu với Python — Báo cáo Cuối Kỳ"
author: "Nguyen Dinh Anh Dung (dng-nguyn)"
date: "Tháng 6, 2026"
---

\newpage

# Mục lục

1. Giới thiệu
2. Tổng quan tài liệu
3. Thu thập và parse dữ liệu
4. Tiền xử lý dữ liệu
5. Phân tích khám phá dữ liệu (EDA)
6. Xây dựng mô hình
7. Tối ưu hóa mô hình và Đặc trưng bên ngoài
8. Dự báo và Đánh giá
9. Kết luận
10. Tài liệu tham khảo

\newpage

# Giới thiệu

## Bối cảnh

Du lịch là một trong những ngành kinh tế quan trọng nhất của Việt Nam, đóng góp khoảng 7\% GDP và hỗ trợ khoảng 5,96 triệu việc làm tính đến năm 2024 trên cả nước trong những năm gần đây [1]. Việt Nam đã chào đón lượng khách kỷ lục 18,0 triệu lượt khách quốc tế vào năm 2019, xếp thứ năm ở khu vực Châu Á - Thái Bình Dương (thứ tư ở Đông Nam Á theo dữ liệu PATA nửa đầu năm 2019) [2]. Đại dịch COVID-19 đã khiến lượng khách quốc tế giảm 78,7\% vào năm 2020 [3], với các biên giới bị đóng cửa hiệu quả từ tháng 4 năm 2020 đến tháng 3 năm 2022 (Nghị quyết số 32/NQ-CP). Ngành du lịch Việt Nam sau đó đã phục hồi mạnh mẽ, đạt mức cao kỷ lục mọi thời đại là 21,2 triệu lượt khách vào năm 2025 [4].

Nghiên cứu này phân tích lượng khách du lịch quốc tế hàng tháng đến Việt Nam từ 32 quốc gia nguồn trong giai đoạn 2008--2026 bằng cách sử dụng dữ liệu do Tổng cục Thống kê (GSO) công bố. Độ phân giải hàng tháng cung cấp số lượng quan sát nhiều gấp khoảng 3 lần so với tổng hợp theo quý, cho phép suy luận thống kê mạnh mẽ hơn và độ phân giải mùa vụ chi tiết hơn.

## Mục tiêu

1. Phân tích xu hướng, tính mùa vụ và cơ cấu quốc gia nguồn của lượng khách quốc tế đến Việt Nam
2. So sánh các mô hình dự báo: Hồi quy tuyến tính (Linear Regression), Random Forest, XGBoost, SARIMAX, mô hình nền tảng Chronos-T5, và mô hình phương trình vi phân ngẫu nhiên CIR\#
3. Đánh giá tác động của các đặc trưng bên ngoài (tỷ giá hối đoái, chính sách thị thực/visa) đến độ chính xác của dự báo
4. Tạo dự báo trước 12 tháng với các khoảng tin cậy
5. Tài liệu hóa các giả định, hạn chế và điều kiện biên của mô hình

## Phạm vi

- **Đối tượng:** Lượng khách du lịch quốc tế hàng tháng đến Việt Nam theo quốc gia nguồn
- **Thời gian:** Tháng 7 năm 2008 -- Tháng 5 năm 2026 (không có dữ liệu cho năm 2021 do đóng cửa biên giới vì COVID-19; chỉ có sẵn dữ liệu từ tháng 1 đến tháng 3 năm 2020)
- **Quốc gia:** 32 quốc gia riêng lẻ (không bao gồm dữ liệu tổng hợp theo khu vực)
- **Đối tượng mục tiêu:** $Y^{32}_t$ = tổng lượng khách từ 32 quốc gia được mô hình hóa. Lưu ý: tổng số chính thức GSO tất cả các thị trường ($Y^{\mathrm{official}}_t$) bao gồm các danh mục còn lại không nằm trong 32 quốc gia, do đó $Y^{\mathrm{official}}_t > Y^{32}_t$ (ví dụ: 18,0 triệu so với 17,5 triệu năm 2019). Tất cả việc điều chỉnh mô hình, xác thực và dự báo sử dụng $Y^{32}_t$.
- **Công cụ:** Python 3, pandas, scikit-learn, XGBoost, statsmodels, Chronos-T5, yfinance

\newpage

# Tổng quan tài liệu

Dự báo nhu cầu du lịch đã được nghiên cứu rộng rãi trong ba thập kỷ qua, tạo ra một phương pháp luận phong phú bao gồm kinh tế lượng, học máy, và gần đây hơn, các mô hình nền tảng (foundation models). Phần này xem xét các truyền thống phương pháp chính, định vị nghiên cứu hiện tại trong bối cảnh văn liệu khu vực về dự báo du lịch Đông Nam Á, và thảo luận các tiến bộ gần đây liên quan đến dự báo tổng hợp hàng tháng.

## Các phương pháp kinh tế lượng nền tảng

Song và Witt [5] cung cấp phương pháp luận nền tảng về mô hình hóa nhu cầu du lịch bằng kinh tế lượng, thiết lập khung dựa trên hồi quy trong đó lượng khách du lịch được mô hình hóa như một hàm của thu nhập, giá tương đối và chi phí vận chuyển. Một đánh giá toàn diện của Song và các cộng sự [6], bao gồm 211 bài báo trên 19 tạp chí, chỉ ra rằng không có phương pháp đơn lẻ nào vượt trội trong mọi bối cảnh dự báo; hiệu suất phụ thuộc quan trọng vào độ chi tiết dữ liệu, khoảng thời gian dự báo, đặc điểm điểm đến và sự hiện diện của các điểm gãy cấu trúc (structural breaks). Phân tích của họ cho thấy các mô hình sửa lỗi (error-correction models) và mô hình tham số thay đổi theo thời gian hoạt động tốt nhất ở tần số năm và quý, trong khi các biến thể SARIMA chiếm ưu thế ở tần số hàng tháng.

Đối với các phương pháp chuỗi thời gian, Hyndman và Athanasopoulos [7] trình bày tài liệu tham khảo tiêu chuẩn cho các mô hình họ ARIMA, bao gồm cả các phần mở rộng mùa vụ (SARIMA/SARIMAX). Phương pháp Box--Jenkins [8] vẫn là khung cổ điển để nhận dạng, ước lượng và kiểm tra chẩn đoán các mô hình ARIMA. Đối với dữ liệu du lịch hàng tháng, chu kỳ mùa vụ $s = 12$ là tiêu chuẩn, và công thức SARIMAX cho phép các hồi quy tử ngoại như biến giả chính sách, tỷ giá hối đoái và chỉ báo khủng hoảng [7]. Tuy nhiên, các mô hình họ ARIMA giả định tính dừng (stationarity) hoặc tính dừng sau khi lấy sai phân, điều này giới hạn khả năng ngoại suy các điểm gãy cấu trúc như quỹ đạo phục hồi hậu COVID-19 được quan sát tại Việt Nam và Đông Nam Á.

## Phương pháp học máy tập hợp

Các phương pháp học máy tập hợp đã ngày càng trở nên nổi bật trong dự báo du lịch từ giữa thập kỷ 2010. Random Forest [9] giảm phương sai thông qua bootstrap aggregation (bagging) của các cây quyết định đã được loại bỏ tương quan, với lấy mẫu đặc trưng ngẫu nhiên tại mỗi nút chia. XGBoost [10] xây dựng các cây một cách tuần tự bằng gradient boosting với regularization rõ ràng, xuất sắc trong việc nắm bắt các tương quan đặc trưng phi tuyến trên dữ liệu dạng bảng. Sự đánh đổi giữa bias và variance [11] giải thích tại sao các mô hình đơn giản hơn (như hồi quy tuyến tính hoặc Random Forest) có thể vượt trội các giải pháp thay thế phức tạp (như XGBoost hoặc deep learning) khi kích thước mẫu bị hạn chế.

Trong các ứng dụng cụ thể của du lịch, nhiều nghiên cứu đã ghi nhận hiệu suất cạnh tranh của Random Forest. Ahmed và cộng sự (2023) so sánh SARIMA, Random Forest và LSTM cho lượng khách du lịch đến Thổ Nhĩ Kỳ, phát hiện rằng Random Forest đạt MAPE thấp nhất trên dữ liệu hàng tháng khi các đặc trưng kinh tế vĩ mô exogenous có sẵn. Assaf và Tsionas (2019) chỉ ra rằng các phương pháp gradient boosting (bao gồm XGBoost) đòi hỏi regularization cẩn thận trên các bộ dữ liệu du lịch nhỏ, thường kém hơn Random Forest khi mẫu huấn luyện dưới 200.

## Dự báo du lịch tại Đông Nam Á

Dự báo du lịch cho các điểm đến Đông Nam Á đã nhận được sự chú ý ngày càng tăng khi khu vực nổi lên như một trong những vùng du lịch tăng trưởng nhanh nhất thế giới. Đối với Thái Lan, thị trường du lịch Đông Nam Á lớn nhất, Chan và cộng sự (2021) áp dụng SARIMAX và mô hình mạng nơ-ron cho lượng khách hàng tháng từ 15 quốc gia nguồn, phát hiện rằng các biến exogenous (đặc biệt tỷ giá hối đoái và chỉ báo khủng hoảng) cải thiện dự báo 8--15\% theo MAPE. Su và Lin (2023) so sánh các mô hình lai EMD-ARIMA và ARIMA thuần cho lượng khách hàng tháng của Thái Lan, báo cáo MAPE 5--8\% cho dự báo một bước nhưng 15--25\% cho khoảng dự báo 12 tháng.

Đối với Malaysia, Habibah và cộng sự (2022) áp dụng SARIMA và Facebook Prophet cho lượng khách theo quý, phát hiện hiệu suất tương đương (MAPE 6--9\%). Họ nhận thấy rằng khả năng phát hiện điểm thay đổi tự động của Prophet xử lý giai đoạn khủng hoảng MH370/MH17 năm 2015 tốt hơn SARIMA. Đối với Indonesia, Wijaya và cộng sự (2023) sử dụng ARIMAX với Google Trends như một biến dự báo exogenous cho lượng khách hàng tháng, đạt MAPE 4--7\% trên dữ liệu tiền COVID nhưng ghi nhận sự suy giảm đáng kể hậu COVID.

Nhiều nghiên cứu đa quốc gia cung cấp bối cảnh so sánh. Li và cộng sự (2020) đánh giá SARIMA, Random Forest và LSTM trên năm điểm đến ASEAN (Thái Lan, Malaysia, Singapore, Philippines, Indonesia) sử dụng dữ liệu hàng tháng 2004--2019. Họ phát hiện rằng Random Forest liên tục đạt MAPE thấp nhất (4--8\%) trên các điểm đến. Wu và cộng sự (2024) chỉ ra rằng các mô hình huấn luyện hoàn toàn trên dữ liệu trước năm 2020 đánh giá thấp một cách có hệ thống lượng khách 2022--2023 trên tất cả các điểm đến ASEAN.

Đối với Việt Nam, văn liệu dự báo còn thưa thớt so với tầm quan trọng kinh tế du lịch. Nguyen và Nguyen (2020) áp dụng SARIMA cho lượng khách theo quý 2000--2019, đạt MAPE 8--12\% trên dự báo kiểm tra một bước. Nghiên cứu hiện tại mở rộng các nỗ lực này bằng cách sử dụng dữ liệu hàng tháng trên 32 quốc gia nguồn với sáu nhóm mô hình riêng biệt.

## Deep learning và Mô hình nền tảng cho chuỗi thời gian

Giai đoạn sau năm 2023 đã chứng kiến những tiến bộ nhanh chóng trong dự báo chuỗi thời gian được thúc đẩy bởi các kiến trúc deep learning. Mạng LSTM và các biến thể (ví dụ: LSTM hai chiều, LSTM tăng cường attention) đã được áp dụng rộng rãi cho dữ liệu du lịch, thường yêu cầu 150+ mẫu huấn luyện hàng tháng để vượt trội các phương pháp cổ điển. Các kiến trúc dựa trên Transformer đã nổi lên như những ứng cử viên mạnh mẽ. PatchTST (Nie và cộng sự, 2023) áp dụng chia chuỗi thời gian thành các token theo kênh độc lập, cho phép self-attention hiệu quả trên các chuỗi dài.

Các mô hình nền tảng cho chuỗi thời gian đại diện cho một sự thay đổi mô hình gần đây. Chronos [12], được phát triển bởi Amazon, huấn luyện trước các mô hình dựa trên Transformer trên hàng triệu chuỗi thời gian sử dụng kiến trúc T5 được điều chỉnh cho các giá trị số được mã hóa hóa (tokenized). Không giống như các mô hình cụ thể cho từng tác vụ, Chronos tạo ra các dự báo xác suất thông qua lấy mẫu tự hồi quy mà không cần bất kỳ huấn luyện nào trên bộ dữ liệu mục tiêu. Chronos-t5-tiny (8,4 triệu tham số) sử dụng cửa sổ ngữ cảnh 512 token, phù hợp với cửa sổ huấn luyện 132 tháng được sử dụng ở đây. Phân tích mở rộng quy mô phát hiện rằng biến thể nhỏ nhất (tiny) vượt trội cả small (46M) và base (201M) trên bộ dữ liệu này.

## Mô hình phương trình vi phân ngẫu nhiên

Các mô hình phương trình vi phân ngẫu nhiên (SDE) đã được áp dụng vào dữ liệu du lịch như một giải pháp thay thế cho các phương pháp thời gian rời rạc. Orlando và Bufalo [13, 14] đề xuất mô hình CIR\#, mở rộng quy trình Cox--Ingersoll--Ross với các phần dư được lọc bằng ARIMA, đạt MAPE 1,18\% trên dữ liệu du lịch hàng tháng của Ý (288 quan sát). Phần mở rộng CIR\# thay thế chuyển động Brown bằng các phần dư được lọc bằng ARIMA và chia dữ liệu thành các mẫu phụ xung quanh các điểm gãy cấu trúc. Tuy nhiên, sự thành công của mô hình phụ thuộc vào việc dữ liệu thỏa mãn các giả định hồi quy về giá trị trung bình.

## Lựa chọn biến exogenous

Việc lựa chọn biến exogenous trong các mô hình nhu cầu du lịch là một vấn đề được nghiên cứu kỹ lưỡng. Các yếu tố kinh tế truyền thống bao gồm GDP quốc gia nguồn (đại diện cho sức mua), chỉ số giá tương đối (tỷ lệ CPI giữa điểm xuất phát và điểm đến) và chi phí vận chuyển. Các nghiên cứu gần đây đã tích hợp dữ liệu dấu vết kỹ thuật số: khối lượng tìm kiếm Google Trends, cảm xúc mạng xã hội và dữ liệu đặt vé máy bay. Đối với các điểm đến nhạy cảm với chính sách như Việt Nam, các chỉ báo chính sách thị thực đặc biệt quan trọng: việc giới thiệu e-visa (tháng 2 năm 2017, mở rộng tháng 8 năm 2023) và miễn thị thực đơn phương đã có tác động rõ ràng đến nhu cầu từ các quốc gia được miễn.

## Dữ liệu khuyết thiếu trong chuỗi du lịch

Dữ liệu khuyết thiếu trong chuỗi du lịch là một thách thức nổi tiếng với những hàm ý quan trọng cho huấn luyện mô hình. Little và Rubin [16] cung cấp khung nền tảng cho phân tích thống kê với dữ liệu khuyết thiếu, phân biệt giữa Khuyết thiếu Hoàn toàn Ngẫu nhiên (MCAR), Khuyết thiếu Ngẫu nhiên (MAR) và Khuyết thiếu Không Ngẫu nhiên (MNAR). Trong bối cảnh du lịch, các giá trị khuyết thiếu thường phát sinh từ các quốc gia chưa được đưa vào khung báo cáo.

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
| 2008 (Tháng 7--Tháng 12) | 6--29 | Không có dữ liệu từ tháng 1--6 hoặc tháng 8 |
| 2009--2011 | 11--13 | Bị giới hạn nghiêm trọng |
| 2012 (Th1--Th11) | 10--11 | Chỉ 10--11 quốc gia cho đến tháng 11 |
| 2012 (Th12)--2017 | 29--30 | Bao phủ ổn định từ tháng 12/2012 |
| 2018--2019 | 31 | Mở rộng nhẹ |
| 2020 (Tháng 1--Tháng 3) | 31 | Chỉ trước COVID-19 |
| 2022--2026 | 29--31 | Phục hồi sau COVID-19 |

Khoảng trống bao phủ này có hai hệ quả quan trọng. Thứ nhất, tổng số liệu tổng hợp cho giai đoạn 2009--2011 thấp hơn thực tế một cách nhân tạo. Thứ hai, tổng số liệu tháng 1--tháng 11 năm 2012 chỉ chứa 10--11 quốc gia; bao phủ ổn định 29--31 quốc gia không bắt đầu cho đến tháng 12 năm 2012 và không hoàn toàn đáng tin cậy cho đến giữa năm 2013. Vì lý do này, tất cả việc mô hình hóa trong báo cáo này bắt đầu từ tháng 1 năm 2013.

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
- **Các khoảng trống quốc gia-năm-tháng:** Đối với phân tích tổng hợp, các tháng-quốc gia bị khuyết thiếu được coi là có 0 lượt khách. Điều này ngầm giả định rằng lượng khách là không đáng kể đối với các quốc gia tham gia khung báo cáo sau này (ví dụ: Ấn Độ từ 2018, Ba Lan từ 2024). Mặc dù giả định này đúng cho giai đoạn điền khuyết thiếu COVID, nó có thể đánh giá thấp lượng khách cho một số quốc gia trong những năm cụ thể [16].

## Chia tập Train/Test

- **Tập huấn luyện (Training set):** Tháng 1 năm 2013 -- Tháng 12 năm 2023 (132 tháng, liên tục). Bắt đầu từ năm 2013 đảm bảo bao phủ ổn định 29--31 quốc gia trong suốt thời kỳ huấn luyện. Dữ liệu huấn luyện năm 2012 bị loại trừ vì tháng 1--tháng 11 năm 2012 chỉ chứa 10--11 quốc gia báo cáo. Tập huấn luyện bao gồm các tháng đóng cửa COVID-19 đã được điền giá trị 0 (tháng 4 năm 2020--tháng 12 năm 2021) với `covid_closed = 1`.
- **Tập kiểm tra (Test set):** Tháng 1 năm 2024 -- Tháng 12 năm 2025 (24 tháng). Giai đoạn phục hồi hoàn toàn sau COVID-19.
- **Loại trừ khỏi huấn luyện:** 2008--2012 (độ bao phủ quốc gia hạn chế hoặc không nhất quán).
- **Khoảng thời gian dự báo (Forecast horizon):** Tháng 1 -- Tháng 12 năm 2026 (12 tháng tiếp theo).

Việc này tạo ra 132 quan sát huấn luyện và 24 quan sát kiểm tra cho phân tích tổng hợp hàng tháng, so với 51 và 10 trong phân tích hàng quý trước đó. Lịch liên tục đảm bảo rằng các đặc trưng trễ (`lag_1`, `lag_12`) bắc cầu khoảng trống COVID-19 một cách chính xác: `lag_1` cho tháng 1 năm 2022 là tháng 12 năm 2021 (= 0), không phải tháng 12 năm 2019.

## Tạo đặc trưng

| Đặc trưng | Mô tả |
|---------|-------------|
| `year` | Năm dương lịch |
| `month` | Tháng (1--12) |
| `time_idx` | Năm + (tháng$-$1)/12, chỉ số thời gian liên tục |
| `lag_1` | Tổng lượt khách của tháng trước |
| `covid_closed` | Chỉ báo nhị phân (1 cho tháng 4 năm 2020--tháng 12 năm 2021, 0 cho các tháng khác) |
| `sin_k`, `cos_k` | Các số hạng mùa vụ Fourier: $\sin(2\pi k \cdot m/12)$, $\cos(2\pi k \cdot m/12)$ với $k=1,2$ |
| `tet_month` | Chỉ báo nhị phân cho tháng nghỉ Tết Nguyên đán (tháng 1 hoặc tháng 2 tùy theo âm lịch) |
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
| `sin_k`, `cos_k` ($k$=1,2) | yes | yes | yes | no | no | no |
| `tet_month` | yes | yes | yes | no | no | no |
| `exchange_rate_*` | opt | opt | opt | no | no | no |
| `visa_*` | opt | opt | opt | no | no | no |
| log-transform target | no | no | no | yes | no | no |
| 2026 forecast method | recursive | recursive | recursive | AR struct | N/A | MC sim |

*Lưu ý: Chronos chỉ sử dụng các giá trị chuỗi thời gian thô. SARIMAX nắm bắt tính mùa vụ thông qua thứ tự mùa vụ $(1,1,1)_{12}$. Các số hạng Fourier và tháng Tết được sử dụng bởi tất cả các mô hình ML có giám sát. Tỷ giá và visa bị loại khỏi dự báo 2026 để tránh rò rỉ dữ liệu.*

# Phân tích khám phá dữ liệu (EDA)

## Xuuyên suốt và bối cảnh phạm vi bao phủ

![Tổng lượt khách hàng năm với biểu đồ chồng số lượng quốc gia. Lưu ý rằng tổng số năm 2009--2011 bị giảm do phạm vi bao phủ hạn chế (11--13 quốc gia).](output/eda_total_trend.png)

Biểu đồ cột hiển thị tổng lượng khách hàng năm (trục trái) với số lượng quốc gia báo cáo được hiển thị chồng lên (trục phải, đường màu đỏ). Các quan sát chính:

- **2009--2011:** Chỉ có 11--13 quốc gia báo cáo, làm cho tổng số liệu thấp hơn thực tế một cách nhân tạo.
- **2012--2019:** Tăng trưởng ổn định; bao phủ đạt 29 quốc gia vào tháng 12 năm 2012 và ổn định ở mức 29--31 từ giữa năm 2013 trở đi. Lượng khách tăng từ 6,8 triệu (2012) lên 18,0 triệu (2019). Lưu ý rằng tổng số liệu tháng 1--tháng 11 năm 2012 chỉ phản ánh 10--11 quốc gia và đánh giá thấp lượng khách thực tế.
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

Phần này mô tả từng mô hình dự báo được áp dụng cho **tổng lượng khách hàng tháng** (tổng hợp trên tất cả các quốc gia). Các mô hình ML có giám sát (Hồi quy tuyến tính, Random Forest, XGBoost) sử dụng các đặc trưng trễ, trượt, lịch, COVID, mùa vụ Fourier, tháng Tết, tỷ giá hối đoái và visa khi áp dụng. SARIMAX được điều chỉnh trên $z_t=\log(y_t+1)$ với `covid_closed` như hồi quy tử ngoại. Chronos-T5-tiny và CIR\# sử dụng chuỗi lượng khách thô mà không có đặc trưng kỹ thuật; xem ma trận sử dụng đặc trưng trong Phần 4.3. Các mô hình được đánh giá trên tập kiểm tra (Tháng 1 năm 2024 -- Tháng 12 năm 2025, 24 tháng) bằng bốn chỉ số:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R$^2$** (Hệ số xác định): Tỷ lệ phương sai được giải thích, tính theo $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{\mathrm{train}})^2}$, trong đó $\bar{y}_{\mathrm{train}}$ là giá trị trung bình của tập huấn luyện. Đây là thông lệ tiêu chuẩn cho chuỗi thời gian: giá trị trung bình tập huấn luyện đóng vai trò baseline ngây thơ để đánh giá dự đoán của mô hình. Giá trị âm cho thấy hiệu suất kém hơn so với việc dự đoán giá trị trung bình của tập huấn luyện [11].

## Hồi quy tuyến tính

Hồi quy tuyến tính giả định một mối quan hệ tuyến tính giữa các đặc trưng $\mathbf{x}$ và biến mục tiêu $y$:

$$y = \mathbf{w}^T \mathbf{x} + b$$

Mô hình tối thiểu hóa tổng bình phương phần dư (Ordinary Least Squares):

$$\min_{\mathbf{w}, b} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

**Các giả định:** Tính tuyến tính, tính độc lập của các phần dư, tính đồng nhất phương sai (homoscedasticity), các sai số phân phối chuẩn. Với 132 mẫu huấn luyện và 7 đặc trưng, tỷ lệ quan sát trên tham số (132:8 $\approx$ 16:1) là đầy đủ.

**Khả năng áp dụng:** Đặc trưng `time_idx` cung cấp một tín hiệu tuyến tính mạnh mẽ cho xu hướng đi lên tổng thể. Với 132 mẫu huấn luyện, sự tinh giản của mô hình là một lợi thế so với các giải pháp thay thế phức tạp [11].

**Hạn chế:** Không thể nắm bắt được các mẫu phi tuyến tính, các chu kỳ mùa vụ nằm ngoài những gì các đặc trưng mã hóa theo tháng cung cấp, hoặc các điểm gãy cấu trúc.

## Hồi quy Random Forest

Random Forest [9] là một tập hợp gồm $T$ cây quyết định, mỗi cây được huấn luyện trên một mẫu bootstrap với việc lấy mẫu đặc trưng ngẫu nhiên:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

Việc trung bình hóa các cây quyết định đã được loại bỏ tương quan giúp giảm phương sai trong khi vẫn duy trì khả năng mô hình hóa phi tuyến tính (bagging).

**Các giả định:** Các mẫu độc lập và phân phối chuẩn (i.i.d.). Không yêu cầu tính dừng (stationarity) hoặc tính tuyến tính.
**Các siêu tham số chính:** `n_estimators` = 200, `max_depth` = 10 (để giới hạn quá khớp trên 132 mẫu), `min_samples_split` = 2.

**Lợi thế so với phân tích hàng quý:** Với 132 mẫu huấn luyện (so với 51 mẫu theo quý), Random Forest có thể xây dựng các cây quyết định đa dạng hơn và có khả năng tổng quát hóa tốt hơn.

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

Mô hình SARIMAX$(1,1,1)(1,1,1)_{12}$ được chọn thông qua tìm kiếm lưới tối thiểu hóa AIC trên các tổ hợp $(p,d,q)(P,D,Q)_{12}$. Mô hình áp dụng sai phân bậc một ở cả cấp độ thông thường và cấp độ mùa vụ, với một số hạng tự hồi quy và một số hạng trung bình trượt ở mỗi cấp độ. Một biến exogenous nhị phân `covid_closed` (1 cho tháng 4 năm 2020--tháng 12 năm 2021, 0 cho các tháng khác) báo hiệu điểm gãy cấu trúc do đóng cửa biên giới, cho phép mô hình học được sự thay đổi mức mà không bị gián đoạn trong lịch. Đối với dự báo 2026, mô hình được điều chỉnh lại trên tất cả dữ liệu qua tháng 12 năm 2025 trước khi tạo dự báo trước 12 tháng.

**Các giả định:** Tính dừng sau khi lấy sai phân; các phần dư là nhiễu trắng. Giai đoạn đóng cửa COVID-19 (tháng 4 năm 2020--tháng 12 năm 2021) được điền giá trị 0 với `covid_closed = 1`, duy trì tính liên tục của lịch để các đặc trưng trễ bắc cầu khoảng trống một cách chính xác.

**Lợi thế của dữ liệu hàng tháng:** Với $s = 12$, mô hình nắm bắt các mẫu mùa vụ chi tiết hơn (ví dụ: kỳ nghỉ Tết vào tháng 1/tháng 2, đỉnh mùa hè vào tháng 7/tháng 8) mà mô hình theo quý ($s = 4$) không thể giải quyết.

## Chronos-T5

Chronos [12] là một họ các mô hình nền tảng dựa trên Transformer được huấn luyện trước cho chuỗi thời gian. Không giống như các mô hình khác học từ 132 mẫu huấn luyện, Chronos không được huấn luyện trên dữ liệu này; nó đã được huấn luyện trước trên hàng triệu chuỗi thời gian từ nhiều lĩnh vực khác nhau. Nó hoạt động bằng cách mã hóa (tokenizing) các giá trị chuỗi thời gian và tạo ra các dự báo xác suất thông qua lấy mẫu tự hồi quy.

| Mô hình | Tham số | Cửa sổ ngữ cảnh |
|-------|-----------|---------------|
| chronos-t5-tiny | 8,4M | 512 tokens |
| chronos-t5-small | 46,2M | 512 tokens |
| chronos-t5-base | 201,4M | 512 tokens |

**Lợi thế của dữ liệu hàng tháng:** Với 132 điểm ngữ cảnh (so với 55 điểm theo quý), mô hình có tín hiệu dày đặc hơn để nhận dạng mẫu. Cửa sổ ngữ cảnh 512 token hoàn toàn nằm trong giới hạn cho phép.

## Mô hình phương trình vi phân ngẫu nhiên CIR\#

Mô hình CIR\# [13, 14] mở rộng phương trình vi phân ngẫu nhiên (SDE) Cox--Ingersoll--Ross để dự báo du lịch với dữ liệu bị gián đoạn:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

trong đó $r(t)$ là lượng khách du lịch tổng hợp hàng tháng, biểu thị dưới dạng số đếm (người), $\kappa > 0$ là tốc độ hồi quy về trung bình (mỗi tháng), $\theta$ là mức trung bình dài hạn (người/tháng), $\sigma$ là tham số độ biến động với đơn vị người/tháng$^{1/2}$, và $dW(t)$ là bước tăng của chuyển động Brown chuẩn. Tỷ lệ của $r(t)$ là số lượng khách hàng tháng thô (không biến đổi log hoặc tỷ lệ lại). Dạng rời rạc với $\Delta t = 1$ tháng, xấp xỉ Euler là $\Delta r_t = \kappa(\theta - r_t) + \sigma\sqrt{r_t}\,\epsilon_t$ với $\epsilon_t \sim \mathcal{N}(0,1)$. Các tham số được ước lượng qua OLS của $\Delta r_t$ trên $r_t$:

$$\Delta r_t = \alpha + \beta r_t + \epsilon_t$$

cho $\hat{\alpha} = \hat{\kappa}\hat{\theta}$, $\hat{\kappa} = -\hat{\beta}$, và $\hat{\theta} = \hat{\alpha}/\hat{\kappa}$ khi $\hat{\kappa} > 0$. Độ biến động được ước lượng từ độ lệch chuẩn của phần dư: $\hat{\sigma} = \text{SD}(\hat{\epsilon}_t) / \sqrt{\bar{r}_t}$.

Orlando và Bufalo [13] báo cáo MAPE là 1,18\% trên dữ liệu du lịch hàng tháng của Ý (288 quan sát), giảm 70\% sai số so với SARIMA và Holt--Winters. Phần mở rộng CIR\# thay thế chuyển động Brown bằng các phần dư được lọc bằng ARIMA và phân chia dữ liệu thành các mẫu phụ xung quanh các điểm gãy cấu trúc.

**Thất bại dự kiến trên dữ liệu Việt Nam:** CIR\# được đưa vào cụ thể để kiểm tra điều kiện biên đã được tài liệu hóa [13, 14]: mô hình giả định các quá trình dừng, hồi quy về giá trị trung bình. Hồi quy phù hợp cho $\hat{\beta} < 0$, ngụ ý $\hat{\kappa} > 0$ (hồi quy về trung bình rõ ràng trong mẫu). Tuy nhiên, du lịch Việt Nam thể hiện xu hướng đi lên mạnh mẽ vi phạm giả định dừng: kiểm tra Augmented Dickey-Fuller trên chuỗi huấn luyện không thể bác bỏ giả thuyết đơn vị gốc ở mức ý nghĩa 5\% ($p = 0{,}079$). Điều này có nghĩa là $\hat{\kappa} > 0$ rõ ràng là một tạo phẩm mẫu hữu hạn thay vì hồi quy về trung bình thực sự. Do đó, mô hình tạo ra các dự báo hồi quy về giá trị trung bình trong mẫu thay vì theo quỹ đạo phục hồi hậu COVID-19.

## So sánh mô hình

![So sánh hiệu suất mô hình (MAE, MAPE, R$^2$).](output/model_comparison.png)

| Mô hình            | MAE       | RMSE      | MAPE   | R²       |
|--------------------|-----------|-----------|--------|----------|
| Random Forest      | 117.361   | 151.813   | 7,69%  | 0,963    |
| Linear Regression  | 143.260   | 164.870   | 9,66%  | 0,956    |
| Chronos-T5-tiny    | 172.046   | 221.354   | 11,16% | 0,921    |
| XGBoost            | 248.413   | 300.611   | 16,10% | 0,854    |
| SARIMAX (log)      | 331.011   | 370.909   | 21,52% | 0,778    |
| CIR#               | 394.559   | 465.267   | 24,98% | 0,650    |

*R² được tính theo giá trị trung bình tập huấn luyện ($\bar{y}_{\mathrm{train}} = 717{.}881$) theo thông lệ tiêu chuẩn chuỗi thời gian. Tất cả các mô hình ML có giám sát bao gồm các số hạng mùa vụ Fourier ($\sin(2\pi k \cdot \text{month}/12)$, $\cos(2\pi k \cdot \text{month}/12)$ với $k=1,2$) và chỉ báo tháng Tết. SARIMAX nắm bắt tính mùa vụ thông qua thứ tự mùa vụ $(1,1,1)_{12}$. Chronos chỉ sử dụng các giá trị chuỗi thời gian thô.*

**Các quan sát chính:**

- **Random Forest đạt sai số thấp nhất** theo cả MAE (117.361) và MAPE (7,69\%). Các số hạng Fourier và đặc trưng lịch Tết đã cải thiện RF từ 8,04\% xuống 7,69\% MAPE, với mức cải thiện lớn nhất cho XGBoost (18,39\% → 16,10\%). Linear Regression đạt R² cao nhất (0,956) và MAPE cạnh tranh (9,66\%).
- **Chronos-T5-tiny** (8,4 triệu tham số) đạt MAPE cạnh tranh 11,16\% chỉ sử dụng giá trị chuỗi thời gian thô (không có đặc trưng kỹ thuật), thể hiện sức mạnh của học chuyển tiếp cho dự báo du lịch. Phân tích mở rộng quy mô trên các biến thể tiny/small/base phát hiện rằng mô hình nhỏ nhất hoạt động tốt nhất trên bộ dữ liệu này: tiny (MAPE 11,16\%), base (13,20\%), small (13,89\%).
- **XGBoost kém hơn** Random Forest (MAPE 16,10\% so với 7,69\%), phù hợp với độ nhạy cảm đã biết của gradient boosting với kích thước mẫu nhỏ [10, 11]. Phân tích đường cong học tập xác nhận hành vi không đơn điệu: MAPE dao động từ 17,6\% (50\% dữ liệu) đến 34,2\% (75\%) đến 18,4\% (100\%).
- **Mức độ quan trọng đặc trưng** (Random Forest): `lag_1` chiếm ưu thế, tiếp theo là `lag_12` và `time_idx`. Các số hạng Fourier xếp dưới các đặc trưng trễ nhưng trên `year`.

- **Kết quả đánh giá đa bước.** Đánh giá gốc lăn (rolling-origin) với các khoảng dự báo $h = 1, 3, 6, 12$ xác nhận rằng Random Forest duy trì độ chính xác đáng chú ý ổn định qua các khoảng dự báo (MAPE $\approx$ 9--10\%), trong khi Hồi quy tuyến tính và XGBoost suy giảm đáng kể ở các khoảng dài hơn (MAPE $\approx$ 30--31\% ở $h = 12$).

- **SARIMAX** (MAPE = 21,52\%) với mục tiêu biến đổi log $z_t = \\log(y_t+1)$ và biến exogenous `covid_closed`. Biến đổi log ổn định phương sai và đảm bảo biến đổi ngược không âm, nhưng mô hình vẫn gặp khó khăn trong việc ngoại suy xu hướng sau COVID-19 từ khung quá trình dừng.
- **CIR#** (MAPE = 24,98\%) được đưa vào để kiểm tra điều kiện biên. $\hat{\kappa}$ ước lượng dương, nhưng dữ liệu có xu hướng mạnh vi phạm giả định dừng ($p = 0{,}079$ trên kiểm tra ADF), tạo ra các dự báo hồi quy về trung bình không nắm bắt được tăng trưởng hậu COVID.
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

## Các số hạng mùa vụ Fourier và hiệu ứng lịch Tết

Để mô hình hóa rõ ràng tính mùa vụ ngoài đặc trưng số nguyên `month`, chúng tôi đã thêm các số hạng Fourier (hai hài hòa, $k=1,2$):

$$f_k^{(\sin)}(t) = \sin\!\left(\frac{2\pi k \cdot m_t}{12}\right), \qquad f_k^{(\cos)}(t) = \cos\!\left(\frac{2\pi k \cdot m_t}{12}\right)$$

trong đó $m_t \in \{1,\ldots,12\}$ là tháng dương lịch. Bốn đặc trưng này nắm bắt chu kỳ mùa vụ chính và hài hòa đầu tiên của nó.

Ngoài ra, một chỉ báo nhị phân `tet_month` được thêm vào để nắm bắt hiệu ứng nghỉ Tết Nguyên đán (Tết), dao động giữa tháng 1 và tháng 2 tùy theo âm lịch. Tết rơi vào tháng 1 cho các năm 2012, 2014, 2017, 2020, 2023 và vào tháng 2 cho các năm 2013, 2015, 2016, 2018, 2019, 2021, 2022, 2024, 2025, 2026.

**Tác động đến độ chính xác mô hình:**

| Mô hình | MAPE cơ bản | +Fourier+Tet MAPE | Cải thiện |
|-------|-----------|-------------------|-------------|
| Linear Regression | 9,73% | 9,66% | $-$0,07 pp |
| Random Forest | 8,04% | 7,69% | $-$0,35 pp |
| XGBoost | 18,39% | 16,10% | $-$2,29 pp |

Các số hạng Fourier và chỉ báo Tet mang lại cải thiện lớn nhất cho XGBoost (2,29 phần trăm), cho thấy rằng các đặc trưng mùa vụ rõ ràng giúp gradient boosting nắm bắt các mẫu mà nếu không sẽ đòi hỏi cây sâu hơn. SARIMAX nắm bắt tính mùa vụ thông qua thứ tự mùa vụ $(1,1,1)_{12}$ và không hưởng lợi từ các số hạng Fourier.

## Phân tích hành vi mở rộng quy mô Chronos

Chúng tôi đánh giá ba biến thể Chronos-T5 với kích thước tăng dần trên cùng tập kiểm tra để đặc trưng hóa hành vi mở rộng quy mô:

| Biến thể | Tham số | MAPE | R² |
|---------|-----------|------|------|
| chronos-t5-tiny | 8,4M | 11,16% | 0,921 |
| chronos-t5-small | 46,2M | 13,89% | 0,880 |
| chronos-t5-base | 201,4M | 13,20% | 0,926 |

Mô hình nhỏ nhất (tiny) đạt MAPE tốt nhất, trong khi mô hình lớn nhất (base) đạt R² tốt nhất. Kết quả ngược trực giác này — các mô hình lớn hơn không cải thiện độ chính xác một cách có hệ thống — phù hợp với ngữ cảnh huấn luyện hạn chế (132 tháng): cửa sổ ngữ cảnh 512 token của cả ba biến thể có thể chứa toàn bộ lịch sử huấn luyện, nên các tham số bổ sung trong mô hình lớn hơn không cung cấp thêm thông tin.

## Phân tích đường cong học tập XGBoost

Để điều tra hiệu suất kém của XGBoost, chúng tôi đánh giá MAPE ở các tỷ lệ mẫu huấn luyện khác nhau:

| Tỷ lệ huấn luyện | N mẫu | MAPE |
|-------------------|-----------|------|
| 25% | 33 | 58,01% |
| 50% | 66 | 17,61% |
| 75% | 99 | 34,15% |
| 100% | 132 | 18,39% |

Hành vi không đơn điệu (75\% dữ liệu cho MAPE tệ hơn 50\%) xác nhận rằng XGBoost thể hiện phương sai cao trên bộ dữ liệu này. Đây là đặc trưng của gradient boosting với dữ liệu huấn luyện hạn chế: cơ chế sửa sai tuần tự khuếch đại nhiễu khi kích thước mẫu không đủ để ước lượng phần dư một cách đáng tin cậy. Đường cong học tập không cho thấy sự hội tụ rõ ràng, cho thấy XGBoost sẽ đòi hỏi dữ liệu huấn luyện nhiều hơn đáng kể để đạt được hiệu suất của Random Forest.

\newpage

# Dự báo và Đánh giá

## Dự đoán so với Thực tế (Tập kiểm tra)

![Lượng khách hàng tháng dự đoán so với thực tế cho tập kiểm tra (2024--2025).](output/pred_vs_actual.png)

XGBoost và Random Forest theo dõi giá trị thực tế chặt chẽ nhất, trong khi SARIMAX và CIR\# phân kỳ đáng kể. Chronos-T5-tiny cung cấp dự báo zero-shot cạnh tranh tốt nhất (MAPE = 11,16\%). Các mô hình dựa trên cây (RF, XGBoost) cho thấy sự biến động nhiều hơn trong các dự đoán, trong khi Hồi quy tuyến tính tạo ra các dự báo mượt mà hơn.

**Ghi chú phương pháp dự báo.** Tất cả bốn mô hình tạo dự báo trước 12 tháng cho năm 2026. Các mô hình dựa trên cây (Hồi quy tuyến tính, Random Forest, XGBoost) sử dụng chiến lược đệ quy: dự báo của mỗi tháng được đưa trở lại làm đặc trưng `lag_1` cho tháng tiếp theo. Cách tiếp cận này tích lũy sai số tại mỗi bước nhưng nắm bắt được các mẫu phi tuyến mà các mô hình cây đã học. SARIMAX sử dụng cấu trúc tự hồi quy để tạo dự báo đa bước trực tiếp. Giá trị trung bình tập hợp (hiển thị bằng màu đỏ) lấy trung bình của cả bốn mô hình, cung cấp dự báo mạnh mẽ hơn bất kỳ mô hình đơn lẻ nào. Dải bóng hiển thị phạm vi giữa mô hình lạc quan nhất và bi quan nhất — thước đo dễ hiểu hơn khoảng tin cậy SARIMAX. Hơn nữa, các đặc trưng bên ngoài động (như tỷ giá hối đoái) đã bị loại khỏi các mô hình dự báo ngoài mẫu cuối cùng năm 2026 để tránh rò rỉ dữ liệu, nghĩa là các dự báo đa bước dựa hoàn toàn vào các mẫu tự hồi quy, chỉ số lịch và chỉ báo chính sách.

## Dự báo SARIMAX 12 tháng (2026)

![Dự báo 12 tháng của SARIMAX cho năm 2026 với khoảng tin cậy 95\%.](output/forecast_plot.png)

*Hình hiển thị dự báo chỉ-SARIMAX (điều chỉnh lại qua tháng 12 năm 2025) với khoảng tin cậy 95\%. Dự báo điểm tập hợp được báo cáo riêng bên dưới.*

| Tháng | Dự báo | KTC 95\% Dưới | KTC 95\% Trên |
|-------|--------|----------------|----------------|
| Tháng 1/2026 | 1.769.400 | 1.065.283 | 2.938.916 |
| Tháng 2/2026 | 1.851.450 | 803.824 | 4.264.449 |
| Tháng 3/2026 | 1.647.989 | 542.942 | 5.002.129 |
| Tháng 4/2026 | 1.540.470 | 400.224 | 5.929.291 |
| Tháng 5/2026 | 1.499.244 | 316.193 | 7.108.713 |
| Tháng 6/2026 | 1.479.019 | 258.836 | 8.451.254 |
| Tháng 7/2026 | 1.609.229 | 237.686 | 10.895.052 |
| Tháng 8/2026 | 1.827.948 | 230.939 | 14.468.645 |
| Tháng 9/2026 | 1.676.192 | 183.104 | 15.344.327 |
| Tháng 10/2026 | 1.729.753 | 164.843 | 18.150.715 |
| Tháng 11/2026 | 1.910.191 | 160.010 | 22.803.647 |
| Tháng 12/2026 | 1.994.000 | 147.774 | 26.905.886 |

Để ngăn chặn dự báo các lượt khách đến âm không thể có về mặt vật lý và ổn định phương sai, biến mục tiêu đã được biến đổi log ($\log(y+1)$) trước khi điều chỉnh SARIMAX. Các dự báo và khoảng tin cậy kết quả đã được chuyển đổi ngược lại về thang đo gốc bằng $\exp(\cdot) - 1$, tự nhiên giới hạn các cận dưới khoảng tin cậy ở mức 0 mà không cần cắt thủ công tùy ý. Cách tiếp cận này tạo ra các khoảng tin cậy rộng hơn so với mô hình thang đo gốc, phản ánh sự không đối xứng vốn có trong các quy trình nhân.

## Dự báo theo quốc gia nguồn (2026)


**Lưu ý về biến đổi ngược:** Biến đổi ngược $\exp(\hat{z}_t) - 1$ tạo ra một dự báo theo thang đo trung vị trên thang đo gốc, không phải là giá trị trung bình có điều kiện. Dự báo trung bình không lệch sẽ cần một sửa đổi lognorm $\exp(\hat{z}_t + \hat{\sigma}^2_t/2) - 1$ hoặc xử lý dựa trên mô phỏng. Khoảng tin cậy là không đối xứng trên thang đo gốc, phản ánh bản chất nhân của quá trình log-biến đổi.

Các mô hình tập hợp (trung bình của Hồi quy tuyến tính, Random Forest, XGBoost và SARIMAX) được điều chỉnh cho 5 quốc gia nguồn hàng đầu. Các mô hình cây sử dụng dự báo đệ quy để tạo dự đoán trước 12 tháng; SARIMAX sử dụng cấu trúc tự hồi quy trực tiếp.

| Tháng | Hàn Quốc | Trung Quốc | Đài Loan | Nhật Bản | Hoa Kỳ |
|-------|-------|-------|-------|-------|-------|
| Tháng 1/2026 | 415.589 | 253.357 | 90.615 | 64.534 | 76.411 |
| Tháng 2/2026 | 415.133 | 245.562 | 102.170 | 63.645 | 70.282 |
| Tháng 3/2026 | 364.367 | 213.994 | 87.645 | 67.327 | 59.654 |
| Tháng 4/2026 | 365.820 | 223.717 | 92.212 | 62.500 | 58.413 |
| Tháng 5/2026 | 362.484 | 221.843 | 91.027 | 64.439 | 52.122 |
| Tháng 6/2026 | 374.440 | 212.771 | 93.809 | 60.466 | 64.743 |
| Tháng 7/2026 | 380.907 | 225.928 | 102.468 | 64.545 | 68.922 |
| Tháng 8/2026 | 434.960 | 246.152 | 99.805 | 80.036 | 61.241 |
| Tháng 9/2026 | 396.678 | 238.087 | 97.347 | 79.496 | 55.602 |
| Tháng 10/2026 | 404.532 | 239.295 | 101.391 | 71.685 | 58.810 |
| Tháng 11/2026 | 406.612 | 252.036 | 102.543 | 75.630 | 65.001 |
| Tháng 12/2026 | 438.882 | 249.132 | 103.948 | 77.370 | 65.566 |

*Bảng hiển thị giá trị trung bình tập hợp của Hồi quy tuyến tính, Random Forest, XGBoost và SARIMAX. Phạm vi bất đồng mô hình: 8,2M–10,1M tổng.*

5 quốc gia hàng đầu chiếm 73,7\% tổng dự báo tập hợp năm 2026 (10,3 trên 14,0 triệu). Hàn Quốc được dự báo vẫn là thị trường nguồn lớn nhất, tiếp theo là Trung Quốc. Dải bóng trong hình hiển thị phạm vi giữa dự báo cao nhất và thấp nhất của các mô hình — thước đo dễ hiểu hơn khoảng tin cậy SARIMAX.


\newpage

## Kiểm chứng dự báo (Tháng 1–5/2026)

Với dữ liệu thực tế có sẵn cho 5 tháng đầu năm 2026, chúng tôi đánh giá dự báo chỉ-SARIMAX (điều chỉnh lại qua tháng 12 năm 2025) so với tổng 32 quốc gia ($Y^{32}_t$) từ `df_monthly.csv`. Công thức MAPE sử dụng là:

$$\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

![Dự báo so với thực tế cho tháng 1—5/2026 (tổng hợp và các quốc gia nguồn hàng đầu).](output/forecast_validation.png)

| Tháng | Thực tế ($Y^{32}_t$) | Dự báo SARIMAX | Sai số | Sai số % tuyệt đối |
|-------|---------|--------|--------|--------|
| Tháng 1/2026 | 2.210.922 | 1.769.400 | $-$20,0% | 20,0% |
| Tháng 2/2026 | 1.970.499 | 1.851.450 | $-$6,0% | 6,0% |
| Tháng 3/2026 | 1.877.290 | 1.647.989 | $-$12,2% | 12,2% |
| Tháng 4/2026 | 1.828.573 | 1.540.470 | $-$15,8% | 15,8% |
| Tháng 5/2026 | 1.606.802 | 1.499.244 | $-$6,7% | 6,7% |
| **MAPE** | | | | **12,1%** |

*Thực tế là $Y^{32}_t$ (tổng 32 quốc gia từ `df_monthly.csv`). Nguồn dự báo: SARIMAX(1,1,1)(1,1,1)$_{12}$ điều chỉnh lại trên dữ liệu huấn luyện 2013--2025, sau đó dự báo 12 bước. Lưu ý: tổng thị trường chính thức GSO ($Y^{\mathrm{official}}_t$) bao gồm các danh mục còn lại không nằm trong 32 quốc gia và sẽ cao hơn.*

**Kiểm chứng theo quốc gia:**

| Quốc gia | MAPE | Ghi chú |
|----------|------|---------|
| Hàn Quốc (South Korea) | 8,3% | Phù hợp tốt nhất; mô hình nắm bắt mẫu mùa vụ tốt |
| Trung Quốc | 48,1% | Đánh giá thấp nhất quán; tăng trưởng cấu trúc từ 2024 không được nắm bắt |
| Nhật Bản | 25,9% | Cải thiện với trình parse đã sửa; mẫu mùa vụ được nắm bắt một phần |
| Campuchia | 50,3% | Thay đổi chế độ biên giới: miễn thị thực + đường bay mới + du lịch y tế; tháng 1/2026 đạt 223.000 (3$\times$ tháng 12/2025) |

**Nghiên cứu điển hình Nga.** Thành phần SARIMAX dự báo 7.000--12.000 khách/tháng cho Nga vào năm 2026 (giá trị trung bình tập hợp trong bảng cao hơn vì ba mô hình khác kéo trung bình lên); số liệu thực tế là 113.000--137.000 --- sai số 10$\times$. Đây là một thất bại dự báo phân phối ngoài (out-of-distribution) do thay đổi chế độ địa chính trị không được đại diện trong dữ liệu huấn luyện 2013--2023. Chiến tranh Nga-Ukraine (2022) đã đóng cửa các điểm đến EU và Hoa Kỳ cho du khách Nga. Việt Nam, với miễn thị thực 45 ngày, khôi phục đường bay thẳng (Aeroflot, VietJet, Vietnam Airlines), và giá cả phải chăng, đã trở thành lựa chọn thay thế chính.

**Nghiên cứu điển hình Campuchia.** Mô hình dự báo 27.000--38.000 khách Campuchia/tháng cho năm 2026; số liệu thực tế dao động từ 53.000 đến 223.000. Hai hiệu ứng cộng hưởng: (1) mức cơ bản tăng từ ~38.000 (2024) lên ~57.000 (2025--2026) do miễn thị thực qua lại, đường bay mới Air Cambodia (~10 chuyến bay/ngày đến Việt Nam), và du lịch y tế xuyên biên giới; (2) đỉnh mùa vụ Tết: tháng 1/2025 đạt 100.000 (+64\%), tháng 1/2026 đạt 223.025 (+206\% so với tháng 12).

MAPE năm tháng 12,1\% cho thấy mô hình SARIMAX đã sửa (điều chỉnh lại qua tháng 12 năm 2025) đạt độ chính xác tốt hơn đáng kể so với ước tính trước đó. Hàn Quốc là quốc gia được dự báo tốt nhất (MAPE = 8,3\%), vì quỹ đạo tăng trưởng của nó gần nhất với phân phối huấn luyện.


# Kết luận

## Kết quả chính

1. **Dữ liệu hàng tháng cải thiện đáng kể phân tích.** Với 132 quan sát huấn luyện (so với 51 quan sát theo quý), các mô hình học máy có lượng dữ liệu gấp đôi để học các mẫu dữ liệu. SARIMAX với $s = 12$ giải quyết cấu trúc mùa vụ chi tiết hơn (kỳ nghỉ Tết, đỉnh mùa hè) mà $s = 4$ không thể nắm bắt.

2. **Sai lệch bao phủ vẫn tồn tại.** Chỉ có 11--13 quốc gia báo cáo dữ liệu hàng tháng trong giai đoạn 2009--2011, so với 29--31 từ tháng 12/2012 trở đi. Các phân tích xu hướng tổng hợp nên sử dụng tháng 1 năm 2013 làm thời điểm bắt đầu để đảm bảo bao phủ quốc gia ổn định trong suốt thời kỳ huấn luyện.

3. **Trung Quốc và Hàn Quốc thống trị** thị trường du lịch, chiếm lượng khách tích lũy lớn nhất. Các thị trường mới nổi (Hồng Kông, Tây Ban Nha, Ý, Philippines) tăng trưởng nhanh nhất trong giai đoạn 2013--2019.

4. **Tính mùa vụ mạnh mẽ:** Tháng 1--Tháng 2 liên tục có lượng khách cao nhất (Tết Nguyên đán + du lịch mùa đông), với đỉnh mùa hè phụ vào tháng 6--Tháng 8.

5. **Điểm gãy cấu trúc sau COVID-19** vẫn là thách thức thống trị. Khi R² được tính theo giá trị trung bình tập huấn luyện (thông lệ tiêu chuẩn chuỗi thời gian), tất cả các mô hình đều đạt R² dương vì các dự đoán nắm bắt xu hướng đi lên từ mức huấn luyện ($\bar{y}_{\mathrm{train}} = 717{.}881$) đến phạm vi tập kiểm tra cao hơn nhiều. Bảng so sánh mô hình đã sửa cho thấy Random Forest đạt MAPE tốt nhất (7,69\%) và MAE (117.361), tiếp theo là Hồi quy tuyến tính (MAPE 9,66\%, R² = 0,956). Chronos-T5-tiny đạt MAPE cạnh tranh (11,16\%) mà không cần huấn luyện cụ thể cho tác vụ, và phân tích mở rộng quy mô phát hiện rằng biến thể nhỏ nhất vượt trội cả small và base trên bộ dữ liệu 132 tháng này. XGBoost (MAPE 16,10\%), SARIMAX (MAPE 21,52\%) và CIR\# (MAPE 24,98\%) kém hơn.

6. **CIR\# thất bại trên dữ liệu có xu hướng.** Mặc dù có dữ liệu hàng tháng như khuyến nghị của Orlando và Bufalo [13], giả định hồi quy về giá trị trung bình của mô hình bị vi phạm bởi quỹ đạo du lịch có xu hướng đi lên của Việt Nam (kiểm tra ADF: $p = 0{,}079$). Đây là một điều kiện biên đã được tài liệu hóa cho mô hình [13, 14].

7. **Các số hạng Fourier và đặc trưng lịch Tết mang lại cải thiện có thể đo lường được.** Việc thêm $\sin(2\pi k \cdot m/12)$ và $\cos(2\pi k \cdot m/12)$ ($k=1,2$) và chỉ báo nhị phân tháng Tết đã giảm MAPE từ 0,07 đến 2,29 phần trăm trên các mô hình có giám sát, với mức cải thiện lớn nhất cho XGBoost (18,39\% → 16,10\%). Tỷ giá hối đoái và chỉ báo chính sách thị thực chỉ mang lại cải thiện bổ sung không đáng kể.

## Hạn chế

1. **Khoảng trống COVID-19 (2020--2021):** Điền giá trị 0 cho giai đoạn đóng cửa COVID-19 duy trì tính liên tục của lịch; biến exogenous `covid_closed` báo hiệu điểm gãy cấu trúc cho SARIMAX. Tuy nhiên, các tháng điền 0 vẫn đại diện cho một sự khác biệt so với quy trình tạo dữ liệu thực sự. Phân tích nhạy cảm loại bỏ các tháng này hoặc sử dụng phương pháp điền thay thế sẽ củng cố các tuyên bố về tính bền vững.
2. **Tập huấn luyện hạn chế (132 tháng):** Bắt đầu từ tháng 1 năm 2013 để đảm bảo bao phủ quốc gia ổn định giảm cửa sổ huấn luyện từ 144 xuống 132 tháng. Mặc dù tốt hơn đáng kể so với 51 điểm hàng quý, con số này vẫn hạn chế độ phức tạp của mô hình so với 288 quan sát hàng tháng được sử dụng trong nghiên cứu CIR\# của Ý [13].
3. **Thiếu các đặc trưng bên ngoài:** GDP của quốc gia nguồn, công suất chuyến bay, lượng tìm kiếm trên Google Trends và giá dầu không được bao gồm. Các yếu tố này có thể cải thiện độ chính xác ngoại suy hậu COVID.
4. **Sự không nhất quán về độ bao phủ (2009--2011):** Chỉ có 11--13 quốc gia mỗi tháng làm giới hạn độ tin cậy của các số liệu thống kê tổng hợp cho những năm đầu. Việc điền khuyết thiếu trước khi tham gia được coi là 0 khách có thể giới thiệu các bước nhảy tăng trưởng nhân tạo khi các quốc gia tham gia khung báo cáo.
5. **Chẩn đoán chính thức đã thực hiện:** Kiểm tra ADF trên chuỗi huấn luyện cho $p = 0{,}079$ (gần không dừng ở mức 5\%); kiểm tra KPSS cho $p = 0{,}076$ (gần dừng ở mức 5\%); chuỗi đã lấy sai phân rõ ràng dừng ($p < 0{,}001$). Kiểm tra Ljung-Box trên phần dư SARIMAX cho thấy không có tương quan tự ở trễ 6 ($p = 0{,}62$) nhưng tương quan tự mùa vụ có ý nghĩa ở trễ 12 ($p = 0{,}002$). Kiểm tra Breusch-Pagan xác nhận phương sai sai số thay đổi ($p = 0{,}002$). Kiểm tra Jarque-Bera xác nhận phần dư không chuẩn ($p < 0{,}001$).
6. **Hạn chế dự báo đa bước đệ quy:** Các dự báo 2026 sử dụng dự đoán đệ quy, trong đó dự báo mỗi tháng được đưa trở lại làm đặc trưng `lag_1` cho tháng tiếp theo. Đánh giá gốc lăn cho thấy tích lũy sai số ở các khoảng dự báo dài hơn: Random Forest ổn định (MAPE ~10\% ở $h = 12$) nhưng Hồi quy tuyến tính và XGBoost suy giảm xuống MAPE ~30\% ở $h = 12$.
7. **Không có chiến lược tập hợp ngoài trung bình:** Dự báo tập hợp sử dụng trung bình số học đơn giản. Stacked generalization, trung bình có trọng số (dựa trên hiệu suất gần đây), hoặc chiến lược lựa chọn mô hình có thể cải thiện độ chính xác.

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
