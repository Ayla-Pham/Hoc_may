# DÀN Ý BÁO CÁO CUỐI KỲ MÔN HỌC MÁY
## Đề tài: Phân loại và phát hiện tin tức giả mạo (Fake News Detection) bằng các thuật toán Học máy kinh điển

---

## Chương 1. Đặt vấn đề
**1.1. Bối cảnh và tính cấp thiết**
- Tin giả lan truyền nhanh trên mạng xã hội, ảnh hưởng đến dư luận và chính trị.
- Kiểm chứng thủ công tốn thời gian, cần công cụ phân loại tự động.

**1.2. Mục tiêu đề tài**
- Xây dựng mô hình phân loại một bài báo tiếng Anh là tin thật (0) hay tin giả (1).
- So sánh 4 thuật toán: Naïve Bayes, Logistic Regression, Decision Tree, AdaBoost.

**1.3. Phạm vi và dữ liệu**
- Bộ dữ liệu *Fake and real news dataset* (Kaggle, Clément Bisaillon): `True.csv` và `Fake.csv`.
- Ngôn ngữ tiếng Anh, chỉ dùng phương pháp Học máy cổ điển (không dùng Deep Learning).

**1.4. Bố cục báo cáo** (tóm tắt các chương sau)

---

## Chương 2. Phân tích dữ liệu và EDA
**2.1. Mô tả dữ liệu**
- Các cột gốc: `title`, `text`, `subject`, `date`.
- Tổng số bài ban đầu: 44.898 (Real 21.417, Fake 23.481). Sau khi bỏ 5.793 bài trùng lặp và vài bài rỗng sau làm sạch còn **39.100 bài**.
- Chỉ dùng `title + text` làm đầu vào. Không dùng `subject` và `date` vì chúng gợi nhãn trực tiếp (nêu lý do).

**2.2. Tiền xử lý văn bản**
- Chữ thường, xóa link, thẻ HTML, dấu câu, chữ số, stopwords tiếng Anh (NLTK).
- Ví dụ một bài trước và sau khi làm sạch.

**2.3. Phân bố nhãn**
- Real: 21.197 bài (54,21%). Fake: 17.903 bài (45,79%). Dữ liệu khá cân bằng.
- *Chèn hình:* `class_distribution.png`.

**2.4. Độ dài bài viết**
- Trung bình 234,4 từ (Real) và 239,8 từ (Fake). Gần như bằng nhau, độ dài không phân biệt được hai lớp.

**2.5. WordCloud**
- *Chèn hình:* `wordcloud_real.png`, `wordcloud_fake.png`.
- Tin thật: `reuters`, `said`, `washington`, `government`, các thứ trong tuần (văn phong tường thuật).
- Tin giả: `featured image`, `getty image`, `video`, `via`, `twitter`, `hillary clinton` (văn phong bình luận, nhiều hình ảnh).
- `trump` xuất hiện ở cả hai lớp nên không phân biệt được.

---

## Chương 3. Kiến trúc phương pháp
**3.1. Sơ đồ tổng quát (pipeline)**
- Dữ liệu thô → làm sạch → chia train/test → TF-IDF → huấn luyện → đánh giá → giải thích mô hình.

**3.2. Chia dữ liệu và tránh rò rỉ dữ liệu**
- `train_test_split(test_size=0.2, random_state=42, stratify=y)`: train 31.280 bài, test 7.820 bài.
- Chia **trước** khi vector hóa. TF-IDF chỉ `fit` trên train, test chỉ `transform`.

**3.3. Vector hóa TF-IDF**
- `TfidfVectorizer(max_features=5000, stop_words='english')`.
- Ma trận thưa, chỉ 2,38% ô khác 0.

**3.4. Bốn mô hình sử dụng** (mỗi mô hình nêu ý tưởng ngắn gọn, công thức chính, tham số điều chỉnh)
- Multinomial Naïve Bayes: tham số làm mượt `alpha`.
- Logistic Regression: tham số regularization `C`.
- Decision Tree: độ sâu tối đa `max_depth`.
- AdaBoost: kết hợp các bộ phân loại yếu (`n_estimators = 100`).

**3.5. Các độ đo đánh giá**
- Confusion Matrix, Accuracy, Precision, Recall, F1-Score, AUC (nêu công thức).

**3.6. K-Fold Cross-Validation (K = 5)**
- Thực hiện trên tập Train, đo bằng F1.
- Dùng `Pipeline` (TF-IDF + mô hình) để mỗi fold chỉ học TF-IDF từ phần train của fold đó.

---

## Chương 4. Kết quả và bảng số liệu so sánh
**4.1. Kết quả chỉnh tham số bằng K-Fold (F1 trung bình, tập Train)**

| Mô hình | Tham số thử | Tham số chọn | F1 tốt nhất |
|---|---|---|---|
| Naïve Bayes | alpha = 0.01 / 0.1 / 0.5 / 1 / 2 | 0.01 | 0,9295 |
| Logistic Regression | C = 0.01 / 0.1 / 1 / 10 / 100 | 100 | 0,9912 |
| Decision Tree | max_depth = 5 / 10 / 20 / 40 | 5 | 0,9951 |
| AdaBoost | n_estimators = 100 | 100 | 0,9950 |

- Nhận xét: C nhỏ (regularization mạnh) làm Logistic Regression underfit (F1 = 0,9140 khi C = 0.01). Cây càng sâu thì F1 càng giảm (dấu hiệu overfit). Độ lệch chuẩn giữa các fold dưới 0,003 nên mô hình ổn định.

**4.2. Kết quả trên tập Test**

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naïve Bayes | 0,9362 | 0,9264 | 0,9349 | 0,9306 | 0,9819 |
| Logistic Regression | 0,9891 | 0,9916 | 0,9846 | 0,9881 | 0,9991 |
| Decision Tree | 0,9962 | 0,9989 | 0,9927 | 0,9958 | 0,9958 |
| AdaBoost | 0,9968 | 0,9980 | 0,9950 | 0,9965 | 0,9993 |

- *Chèn hình:* `confusion_matrices.png`, `roc_curves.png`.
- Mô hình tốt nhất: AdaBoost, chỉ sai 25 bài trên 7.820 bài. Naïve Bayes thấp nhất do giả định các từ độc lập.
- Kết quả K-Fold và kết quả test gần nhau nên không có dấu hiệu overfit.

**4.3. Giải thích mô hình (Feature Importance)**
- *Chèn hình:* `feature_importance.png`.
- Top 10 từ kéo về Fake: video, read, featured, image, gop, breaking, getty, hillary, mr, wire.
- Top 10 từ kéo về Real: reuters, said, washington, tuesday, nov, wednesday, republican, thursday, comment, monday.

**4.4. Thảo luận và hạn chế**
- Từ `reuters` có hệ số −82,6, gấp khoảng 2,5 lần từ đứng thứ hai. Mô hình học nhiều dấu hiệu về **nguồn tin và văn phong** (tên hãng tin, ngày trong tuần, chú thích ảnh) chứ chưa chắc học được **tính xác thực của nội dung**.
- Vì vậy độ chính xác gần 99,7% có thể không giữ nguyên với tin từ nguồn khác.
- Hệ số lớn là do C = 100 (regularization yếu).
- Hướng cải thiện: loại tên nguồn tin khỏi văn bản, kiểm thử trên bộ dữ liệu từ nguồn khác, dùng n-gram.

---

## Chương 5. Kết luận và Demo
**5.1. Demo hàm `predict_news`**
- Quy trình: làm sạch → TF-IDF (chỉ `transform`) → dự đoán nhãn + độ tự tin (%).
- Mẫu 1 (tin kiểu Reuters, "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a bill..."): dự đoán **TIN THẬT**, độ tự tin 100,00%.
- Mẫu 2 (tin giật gân, "BREAKING: SHOCKING VIDEO shows what they are hiding..."): dự đoán **TIN GIẢ**, độ tự tin 100,00%.
- Nhận xét: cả hai mẫu đều đúng, nhưng độ tự tin 100% là quá cao. Nguyên nhân là mô hình dùng C = 100 (regularization yếu) và các từ `reuters`, `breaking`, `video` có hệ số rất lớn. Mẫu 1 có sẵn từ `reuters`, `tuesday`, `washington`. Điều này khớp với phần hạn chế ở mục 4.4: mô hình bám vào dấu hiệu nguồn tin.

**5.2. Kết luận**
- Cả 4 mô hình đều phân loại tốt, AdaBoost cho kết quả cao nhất (Accuracy 99,68%, F1 99,65%).
- Việc chia dữ liệu trước khi TF-IDF, dùng K-Fold và chỉnh tham số giúp kết quả đáng tin cậy.

**5.3. Hướng phát triển**
- Kiểm thử trên dữ liệu ngoài bộ dữ liệu này, thêm đặc trưng n-gram, thử các mô hình khác.

---

## Tài liệu tham khảo
- Bisaillon, C., *Fake and real news dataset*, Kaggle.
- Giáo trình môn Học máy: các chương Phân loại, KNN/Tree/Boosting, Naïve Bayes.
- Tài liệu scikit-learn, NLTK.
