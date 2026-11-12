# ============================================================
# ĐỀ TÀI: PHÂN LOẠI VÀ PHÁT HIỆN TIN TỨC GIẢ MẠO (FAKE NEWS)
# Dataset: Fake and real news dataset (True.csv / Fake.csv)
# Nhãn: 0 = Tin thật (Real), 1 = Tin giả (Fake)
# ============================================================

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)

# Để in tiếng Việt có dấu ra console của Windows không bị lỗi font
sys.stdout.reconfigure(encoding="utf-8")

# Tải danh sách stopwords tiếng Anh của NLTK (chỉ tải ở lần chạy đầu)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


# ------------------------------------------------------------
# BƯỚC 1: Đọc dữ liệu và tiền xử lý văn bản
# ------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()                    # chuyển thành chữ thường
    text = re.sub(r"http\S+", " ", text)        # xóa đường link web
    text = re.sub(r"<.*?>", " ", text)          # xóa thẻ HTML
    text = re.sub(r"[^a-z\s]", " ", text)       # xóa dấu câu và chữ số, chỉ giữ chữ cái
    words = text.split()
    # bỏ stopwords và bỏ luôn các từ chỉ có 1 ký tự
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


def load_data():
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    true_df["label"] = 0    # tin thật
    fake_df["label"] = 1    # tin giả

    # Gộp 2 bảng lại thành 1 bảng duy nhất
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Nối tiêu đề và nội dung thành 1 cột
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Bỏ các bài bị trùng nội dung để tránh cùng 1 bài nằm ở cả train lẫn test
    truoc = len(df)
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    print("Đã bỏ", truoc - len(df), "bài trùng lặp")

    print("Đang làm sạch văn bản, đợi một lát...")
    df["clean"] = df["content"].apply(clean_text)

    # Bỏ những bài sau khi làm sạch bị rỗng
    df = df[df["clean"].str.strip() != ""].reset_index(drop=True)
    return df


# ------------------------------------------------------------
# BƯỚC 2: Khám phá dữ liệu (EDA)
# ------------------------------------------------------------
def explore_data(df):
    print("\n===== EDA =====")
    print("Tổng số bài báo:", len(df))
    print("Số bài theo từng lớp (0 = Real, 1 = Fake):")
    print(df["label"].value_counts())
    print("Tỷ lệ (%):")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    # Biểu đồ cột số lượng 2 lớp
    plt.figure(figsize=(5, 4))
    sns.countplot(x="label", data=df)
    plt.xticks([0, 1], ["Real (0)", "Fake (1)"])
    plt.title("Số lượng bài báo mỗi lớp")
    plt.xlabel("Nhãn")
    plt.ylabel("Số bài")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()

    # Độ dài trung bình (số từ) của bài theo từng lớp
    so_tu = df["clean"].apply(lambda x: len(x.split()))
    print("\nSố từ trung bình mỗi bài (sau khi làm sạch):")
    print(so_tu.groupby(df["label"]).mean().round(1))

    # WordCloud cho tin thật và tin giả
    real_text = " ".join(df[df["label"] == 0]["clean"])
    fake_text = " ".join(df[df["label"] == 1]["clean"])

    wc_real = WordCloud(width=900, height=500, background_color="white",
                        max_words=150).generate(real_text)
    wc_fake = WordCloud(width=900, height=500, background_color="white",
                        max_words=150).generate(fake_text)

    plt.figure(figsize=(8, 4.5))
    plt.imshow(wc_real, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud - Tin thật (Real)")
    plt.tight_layout()
    plt.savefig("wordcloud_real.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.imshow(wc_fake, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud - Tin giả (Fake)")
    plt.tight_layout()
    plt.savefig("wordcloud_fake.png", dpi=150)
    plt.close()
    print("Đã lưu: class_distribution.png, wordcloud_real.png, wordcloud_fake.png")


# ------------------------------------------------------------
# BƯỚC 3: Chia train/test và vector hóa bằng TF-IDF
# ------------------------------------------------------------
def chia_va_vector_hoa(df):
    X = df["clean"]
    y = df["label"]

    # Lấy 80% train và 20% test. Chia TRƯỚC khi TF-IDF để không bị rò rỉ dữ liệu.
    # stratify=y để tỷ lệ tin thật/giả ở train và test giống nhau
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Dùng TF-IDF biến câu chữ thành số
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)   # chỉ fit trên tập train
    X_test_tfidf = tfidf.transform(X_test)         # tập test chỉ transform

    print("\n===== TF-IDF =====")
    print("Kích thước tập train:", X_train_tfidf.shape)
    print("Kích thước tập test :", X_test_tfidf.shape)
    return X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf


# ------------------------------------------------------------
# BƯỚC 4a: K-Fold Cross-Validation để chỉnh tham số
# ------------------------------------------------------------
def tao_pipeline(model):
    # Dùng Pipeline để ở mỗi fold, TF-IDF chỉ học từ phần train của fold đó
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("model", model)
    ])


def chinh_tham_so(X_train, y_train):
    print("\n===== K-FOLD CROSS-VALIDATION (K = 5) trên tập Train =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Thử các giá trị alpha của Naive Bayes (alpha là hệ số làm mượt Laplace)
    print("\n--- Naive Bayes: thử alpha ---")
    best_alpha = None
    best_score = 0
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
        pipe = tao_pipeline(MultinomialNB(alpha=alpha))
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="f1")
        print("alpha =", alpha, "| F1 trung bình = %.4f (+/- %.4f)" % (scores.mean(), scores.std()))
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = alpha
    print("=> Chọn alpha =", best_alpha)

    # Thử các giá trị C của Logistic Regression
    # C nhỏ = regularization mạnh (mô hình đơn giản), C lớn = regularization yếu
    print("\n--- Logistic Regression: thử C ---")
    best_C = None
    best_score = 0
    for C in [0.01, 0.1, 1, 10, 100]:
        pipe = tao_pipeline(LogisticRegression(C=C, max_iter=1000))
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="f1")
        print("C =", C, "| F1 trung bình = %.4f (+/- %.4f)" % (scores.mean(), scores.std()))
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
    print("=> Chọn C =", best_C)

    # Thử độ sâu của Cây quyết định (giới hạn độ sâu để cây không bị overfit)
    print("\n--- Decision Tree: thử max_depth ---")
    best_depth = None
    best_score = 0
    for depth in [5, 10, 20, 40]:
        pipe = tao_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=42))
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="f1")
        print("max_depth =", depth, "| F1 trung bình = %.4f (+/- %.4f)" % (scores.mean(), scores.std()))
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_depth = depth
    print("=> Chọn max_depth =", best_depth)

    # AdaBoost: chỉ kiểm tra độ ổn định với 1 cấu hình (vì chạy khá lâu)
    print("\n--- AdaBoost: n_estimators = 100 ---")
    pipe = tao_pipeline(AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42))
    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="f1")
    print("F1 trung bình = %.4f (+/- %.4f)" % (scores.mean(), scores.std()))

    return best_alpha, best_C, best_depth


# ------------------------------------------------------------
# BƯỚC 4b: Huấn luyện và đánh giá 4 mô hình trên tập Test
# ------------------------------------------------------------
def danh_gia_mo_hinh(models, X_train_tfidf, X_test_tfidf, y_train, y_test):
    print("\n===== KẾT QUẢ TRÊN TẬP TEST =====")
    ket_qua = []
    roc_data = {}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for i, (ten, model) in enumerate(models.items()):
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]    # xác suất là tin giả

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print("\n--------", ten, "--------")
        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1-Score :", round(f1, 4))
        print("AUC      :", round(auc, 4))
        print("Confusion Matrix:")
        print(cm)

        ket_qua.append([ten, acc, prec, rec, f1, auc])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[ten] = (fpr, tpr, auc)

        # Vẽ confusion matrix của từng mô hình
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    ax=axes[i])
        axes[i].set_title(ten)
        axes[i].set_xlabel("Dự đoán")
        axes[i].set_ylabel("Thực tế")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()

    # Vẽ đường ROC của 4 mô hình trên 1 hình
    plt.figure(figsize=(6, 5))
    for ten in roc_data:
        fpr, tpr, auc = roc_data[ten]
        plt.plot(fpr, tpr, label="%s (AUC = %.4f)" % (ten, auc))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Đường cong ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    # Bảng so sánh tổng hợp
    bang = pd.DataFrame(ket_qua, columns=["Mô hình", "Accuracy", "Precision",
                                          "Recall", "F1-Score", "AUC"])
    print("\n===== BẢNG SO SÁNH 4 MÔ HÌNH =====")
    print(bang.round(4).to_string(index=False))
    bang.round(4).to_csv("ket_qua_so_sanh.csv", index=False, encoding="utf-8-sig")
    print("Đã lưu: confusion_matrices.png, roc_curves.png, ket_qua_so_sanh.csv")
    return bang


# ------------------------------------------------------------
# BƯỚC 5: Giải thích mô hình bằng hệ số của Logistic Regression
# ------------------------------------------------------------
def feature_importance(model_lr, tfidf):
    print("\n===== TỪ KHÓA QUYẾT ĐỊNH NHÃN (Logistic Regression) =====")
    ten_tu = np.array(tfidf.get_feature_names_out())
    he_so = model_lr.coef_[0]

    # argsort sắp xếp tăng dần: 10 phần tử cuối là hệ số dương lớn nhất
    top_fake = np.argsort(he_so)[-10:][::-1]
    top_real = np.argsort(he_so)[:10]

    print("\nTop 10 từ kéo về TIN GIẢ (hệ số dương cao nhất):")
    for idx in top_fake:
        print("  %-15s %.4f" % (ten_tu[idx], he_so[idx]))

    print("\nTop 10 từ kéo về TIN THẬT (hệ số âm thấp nhất):")
    for idx in top_real:
        print("  %-15s %.4f" % (ten_tu[idx], he_so[idx]))

    # Vẽ biểu đồ cột ngang cho 20 từ này
    cac_idx = list(top_real) + list(top_fake)
    plt.figure(figsize=(8, 6))
    mau = ["tab:blue"] * 10 + ["tab:red"] * 10
    plt.barh(ten_tu[cac_idx], he_so[cac_idx], color=mau)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 10 từ cho mỗi lớp (xanh = Real, đỏ = Fake)")
    plt.xlabel("Hệ số (coef) của Logistic Regression")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Đã lưu: feature_importance.png")


# ------------------------------------------------------------
# BƯỚC 6: Hàm kiểm tra thử nghiệm thực tế
# ------------------------------------------------------------
def predict_news(news_text, model, tfidf):
    sach = clean_text(news_text)                 # làm sạch văn bản
    vec = tfidf.transform([sach])                # chuyển sang TF-IDF (chỉ transform)
    nhan = model.predict(vec)[0]
    xac_suat = model.predict_proba(vec)[0]       # [P(real), P(fake)]

    if nhan == 1:
        print("Kết quả: TIN GIẢ (Fake) - độ tự tin: %.2f%%" % (xac_suat[1] * 100))
    else:
        print("Kết quả: TIN THẬT (Real) - độ tự tin: %.2f%%" % (xac_suat[0] * 100))
    return nhan


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    # Bước 1: đọc và làm sạch dữ liệu
    df = load_data()

    # Bước 2: EDA
    explore_data(df)

    # Bước 3: chia train/test và TF-IDF
    X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf = chia_va_vector_hoa(df)

    # Bước 4a: K-Fold để chọn tham số
    best_alpha, best_C, best_depth = chinh_tham_so(X_train, y_train)

    # Bước 4b: huấn luyện 4 mô hình với tham số tốt nhất vừa tìm được
    models = {
        "Naive Bayes": MultinomialNB(alpha=best_alpha),
        "Logistic Regression": LogisticRegression(C=best_C, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=best_depth, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    }
    bang = danh_gia_mo_hinh(models, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Bước 5: từ khóa quan trọng
    feature_importance(models["Logistic Regression"], tfidf)

    # Bước 6: demo dự đoán 2 tin thực tế
    print("\n===== DEMO DỰ ĐOÁN =====")
    tin_reuters = ("WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a bill to fund "
                   "the federal government through September, sending the measure to the "
                   "president, who is expected to sign it, officials said. The vote was 79-18.")
    tin_gia = ("BREAKING: SHOCKING VIDEO shows what they are hiding from you! Watch this "
               "unbelievable footage that the mainstream media refuses to show. Share this "
               "before they delete it!!!")

    print("\nTin 1 (kiểu Reuters):")
    predict_news(tin_reuters, models["Logistic Regression"], tfidf)
    print("\nTin 2 (giật gân):")
    predict_news(tin_gia, models["Logistic Regression"], tfidf)
