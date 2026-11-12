# YÊU CẦU DỰ ÁN HỌC MÁY CUỐI KỲ: PHÂN LOẠI VÀ PHÁT HIỆN TIN TỨC GIẢ MẠO (FAKE NEWS DETECTION)

## 1. TỔNG QUAN DỰ ÁN & BỐI CẢNH
- **Đề tài:** Phân loại và phát hiện tin tức giả mạo (Fake News Detection) bằng các thuật toán Học máy kinh điển (Classical Machine Learning).
- **Mục tiêu điểm số:** Điểm 9.0 - 9.5 (Vừa chuẩn phương pháp luận học thuật, vừa giữ đúng phong cách bài làm thực tế của sinh viên đại học).
- **Ngôn ngữ dữ liệu:** Tiếng Anh.
- **Dataset sử dụng:** `Fake and real news dataset` (từ Kaggle của Clément Bisaillon) gồm 2 file CSV đặt cùng thư mục với mã nguồn:
  - `True.csv`: Tin tức thật (gán nhãn `0` / Real).
  - `Fake.csv`: Tin tức giả mạo (gán nhãn `1` / Fake).
  - Các cột gốc: `title`, `text`, `subject`, `date`.

---

## 2. NGUYÊN TẮC QUAN TRỌNG: CODE ĐÚNG PHONG CÁCH SINH VIÊN (TRÁNH BỊ THẦY PHÁT HIỆN DÙNG AI)
- **Hạn chế viết kiểu kỹ sư phần mềm chuyên nghiệp:**
  - Tuyệt đối KHÔNG viết code hướng đối tượng quá phức tạp (Class, Factory pattern, Abstract class, Decorators).
  - Tuyệt đối KHÔNG dùng cú pháp gõ kiểu nghiêm ngặt (Type Hinting kiểu `def clean_text(text: str) -> str:`).
  - Hãy viết code Python tuần tự, rõ ràng, chia theo từng khối hàm (functions) đơn giản, thực dụng đúng với trình độ một sinh viên học môn Học máy trên trường.
- **Lời bình và giải thích (Comments):**
  - Viết chú thích bằng tiếng Việt tự nhiên, giải thích chân phương: dòng này làm gì, biến này mang ý nghĩa gì, tại sao lại dùng tham số đó (ví dụ: `# Dùng TF-IDF biến câu chữ thành số`, `# Lấy 80% train và 20% test`).
  - Tránh các đoạn docstring chuẩn quốc tế kiểu Google/Sphinx dài dòng lộ rõ dấu vết sinh tự động của AI.
- **Về lý thuyết và kỹ thuật:**
  - Chỉ tập trung vào các thuật toán, độ đo có trong giáo trình và đề cương của thầy: Naïve Bayes, Logistic Regression, Decision Tree, AdaBoost, Confusion Matrix, Accuracy, F1-Score, AUC.
  - Không nhồi nhét các mô hình Deep Learning cao siêu (BERT, Transformer, RoBERTa) vì môn học chỉ yêu cầu Học máy cổ điển.

---

## 3. CÁC LỖI HỌC THUẬT CẦN TRÁNH TUYỆT ĐỐI (RÚT KINH NGHIỆM TỪ BÀI BỊ TRỪ ĐIỂM)
1. **Tránh Data Leakage (Rò rỉ dữ liệu):**
   - Phải chia `train_test_split(test_size=0.2, random_state=42)` **TRƯỚC** khi biến đổi dữ liệu văn bản thành số qua TF-IDF.
   - `TfidfVectorizer` chỉ được gọi `.fit_transform()` trên `X_train`, sau đó chỉ gọi `.transform()` trên `X_test`. Tuyệt đối không fit TF-IDF trên toàn bộ dữ liệu trước khi chia.
2. **Tránh dùng tham số mặc định sơ sài:**
   - Trong quá trình đánh giá, hãy sử dụng **K-Fold Cross-Validation (K=5)** trên tập Train để kiểm chứng độ ổn định của mô hình.
   - Thử nghiệm chỉnh vài tham số cơ bản (như tham số `C` trong Logistic Regression, `alpha` trong Naïve Bayes) để chứng minh có hiểu bản chất điều tiết (regularization) của mô hình.
3. **Cần có phần phân tích và minh họa trực quan:**
   - Tạo biểu đồ **WordCloud** so sánh từ khóa của Tin thật và Tin giả.
   - In ra danh sách các từ ngữ có trọng số lớn nhất quyết định nhãn tin giả/tin thật (Model Interpretability / Feature Importance).

---

## 4. CHI TIẾT CÁC BƯỚC THỰC HIỆN TRONG PIPELINE

### Bước 1: Nạp dữ liệu và Tiền xử lý văn bản
- Đọc `True.csv` và `Fake.csv`, tạo cột nhãn `label` (`0` cho True, `1` cho Fake).
- Tạo cột nội dung tổng hợp: `content = title + " " + text`.
- Viết hàm `clean_text(text)` làm sạch câu chữ:
  - Chuyển thành chữ thường (`.lower()`).
  - Dùng regex đơn giản xóa đường link web (`http\S+`), thẻ HTML (`<.*?>`), dấu câu và chữ số.
  - Xóa stopwords tiếng Anh bằng `nltk.corpus.stopwords`.

### Bước 2: Khám phá dữ liệu (EDA)
- In số lượng bài báo của mỗi lớp để xem dữ liệu có cân bằng không.
- Tạo và lưu ảnh **WordCloud** (Đám mây từ khóa):
  - 1 hình cho Tin thật (các từ nổi bật như: reuters, said, president, washington...).
  - 1 hình cho Tin giả (các từ nổi bật như: video, watch, trump, shock, breaking...).
- Lưu biểu đồ ra file ảnh PNG để sinh viên chèn vào bài báo cáo.

### Bước 3: Chia Train/Test & Vector hóa (TF-IDF)
- Chia dữ liệu `X_train, X_test, y_train, y_test` với tỉ lệ 80% - 20%.
- Dùng `TfidfVectorizer(max_features=5000, stop_words='english')`.
- Fit trên `X_train`, transform trên `X_test`.

### Bước 4: Huấn luyện và Đánh giá 4 mô hình quen thuộc
Huấn luyện 4 mô hình:
1. **Multinomial Naïve Bayes (`MultinomialNB`)** - Thuật toán xác suất từ vựng kinh điển.
2. **Hồi quy Logistic (`LogisticRegression`)** - Phân loại tuyến tính, ranh giới rõ ràng.
3. **Cây quyết định (`DecisionTreeClassifier`)** - Phân nhánh điều kiện từ khóa.
4. **Boosting (`AdaBoostClassifier`)** - Kết hợp các bộ phân loại yếu.
- Với mỗi mô hình, in ra màn hình console:
  - `Accuracy` (Độ chính xác)
  - `Precision` (Độ chuẩn xác)
  - `Recall` (Độ gợi nhớ)
  - `F1-Score` (Độ đo cân bằng)
  - `AUC` (Diện tích dưới đường cong ROC)
  - `Confusion Matrix` (Ma trận nhầm lẫn)
### Bước 5: Giải thích mô hình (Feature Importance)
- Dựa vào hệ số `coef_` của Logistic Regression, in ra:
  - Top 10 từ khóa kéo về nhãn TIN GIẢ (hệ số dương cao nhất).
  - Top 10 từ khóa kéo về nhãn TIN THẬT (hệ số âm thấp nhất).

### Bước 6: Hàm kiểm tra thử nghiệm thực tế (Demo)
- Viết hàm `predict_news(news_text)` nhận vào 1 đoạn văn bản tiếng Anh:
  - Làm sạch văn bản \\(\rightarrow\\) Transform qua TF-IDF \\(\rightarrow\\) Dự đoán nhãn + in ra phần trăm độ tự tin (Confidence score %).
  - Cho chạy thử 2 mẫu tin thực tế (1 tin Reuters chính thống, 1 tin giật gân giả mạo).

---

## 5. KẾT QUẢ ĐẦU RA CẦN CUNG CẤP
1. **File mã nguồn Python hoàn chỉnh (`main.py`):** Code chạy mượt mà trên máy tính cá nhân (VS Code), chú thích tiếng Việt tự nhiên, phong cách lập trình chuẩn sinh viên.
2. **Khung dàn ý Báo cáo cuối kỳ (Report Outline):** Dàn ý các chương theo văn phong học thuật sinh viên nộp cho giảng viên bộ môn, gồm: Đặt vấn đề, Phân tích dữ liệu & EDA, Kiến trúc phương pháp, Kết quả & Bảng số liệu so sánh, Kết luận & Demo. 