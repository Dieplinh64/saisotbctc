import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cấu hình tiêu đề trang web
st.title("🌲 Hệ Thống Huấn Luyện Mô Hình Học Máy Random Forest")
st.markdown("---")

# 2. Đọc dữ liệu tài chính (Tự động giả lập dữ liệu đa chiều đầy đủ nhãn nếu chưa có file)
try:
    df = pd.read_csv("financial_anomaly_data.csv")
    # Nếu file gốc chưa có nhãn mục tiêu, ta tự tạo nhãn rủi ro (Label) dựa trên số tiền để mô hình học
    if 'Label' not in df.columns:
        df['Label'] = np.where(df['Amount'] > 80000, 1, 0)
    if 'Frequency' not in df.columns:
        np.random.seed(42)
        df['Frequency'] = np.random.randint(1, 30, size=len(df))
        df['Risk_Factor'] = np.random.uniform(10, 90, size=len(df))
except FileNotFoundError:
    # Tạo dữ liệu kế toán mẫu (300 dòng) phục vụ chạy thử nghiệm
    np.random.seed(42)
    n_samples = 300
    amounts = np.random.exponential(scale=35000, size=n_samples)
    frequencies = np.random.randint(1, 30, size=n_samples)
    risk_factors = np.random.uniform(10, 90, size=n_samples)
    
    df = pd.DataFrame({
        'Amount': amounts,
        'Frequency': frequencies,
        'Risk_Factor': risk_factors
    })
    # Đặt quy luật: Nếu số tiền lớn hoặc hệ số rủi ro cao thì nhãn = 1 (Rủi ro rà soát), ngược lại = 0 (An toàn)
    df['Label'] = np.where((df['Amount'] > 75000) | (df['Risk_Factor'] > 75), 1, 0)

st.subheader("📋 1. Dữ Liệu Nhật Ký Chứng Từ Dùng Để Huấn Luyện")
st.write("Bộ dữ liệu bao gồm các đặc trưng tài chính và cột nhãn mục tiêu `Label` (0: An toàn, 1: Cần kiểm toán rà soát).")
st.dataframe(df.head(8))

# 3. Phân tách biến độc lập (X) và biến phụ thuộc (y)
X = df[['Amount', 'Frequency', 'Risk_Factor']]
y = df['Label']

# 4. Chia tập dữ liệu theo tỷ lệ chuẩn 80% Train, 20% Test như báo cáo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.markdown("---")
st.subheader("⚙️ 2. Cấu Hình Tham Số Mô Hình (Sidebar)")
st.write("Bạn có thể điều chỉnh số lượng cây quyết định trong rừng ngẫu nhiên ở thanh công cụ bên trái.")

# Cho phép tinh chỉnh số lượng cây trên giao diện web
n_estimators = st.sidebar.slider("Số lượng Cây Quyết định (n_estimators):", min_value=10, max_value=200, value=100, step=10)

# --- THỰC HIỆN HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ---
if st.button("🚀 Bắt đầu huấn luyện mô hình Rừng Ngẫu Nhiên"):
    with st.spinner("Hệ thống đang thiết lập các cây quyết định..."):
        # Khởi tạo và huấn luyện mô hình
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Dự đoán thử nghiệm trên tập Test
        y_pred = model.predict(X_test)
        
        # Tính toán độ chính xác
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"🎉 Huấn luyện hoàn tất! Mô hình đạt độ chính xác (Accuracy): **{accuracy * 100:.2f}%**")
        
        # --- HIỂN THỊ ĐỘ QUAN TRỌNG CỦA CÁC THUỘC TÍNH (FEATURE IMPORTANCE) ---
        st.markdown("---")
        st.subheader("📊 3. Đánh Giá Tầm Quan Trọng Của Các Chỉ Số Tài Chính")
        st.write("Biểu đồ thể hiện xem yếu tố nào chi phối nhiều nhất đến quyết định gắn cờ rủi ro của mô hình.")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], palette="viridis", ax=ax)
        plt.title("Độ quan trọng của các thuộc tính đối với mô hình")
        plt.xlabel("Mức độ đóng góp")
        st.pyplot(fig)
        
        # --- HIỂN THỊ MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---
        st.subheader("🧩 4. Ma Trận Nhầm Lẫn (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['An toàn', 'Cần kiểm soát'], 
                    yticklabels=['An toàn', 'Cần kiểm soát'], ax=ax_cm)
        plt.ylabel('Thực tế')
        plt.xlabel('Mô hình dự đoán')
        st.pyplot(fig_cm)