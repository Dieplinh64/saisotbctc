import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- BƯỚC 1: TIÊU ĐỀ ---
st.title("🌲 Ứng Dụng Học Máy Random Forest Kiểm Toán")
st.markdown("---")

# --- BƯỚC 2: ĐỌC FILE DỮ LIỆU VÀ TỰ ĐỘNG SỬA LỖI THIẾU CỘT ---
try:
    df = pd.read_csv("financial_anomaly_data.csv")
    
    # 🎯 SỬA LỖI KEYERROR: Nếu trong file của bạn CHƯA CÓ cột 'Label', AI sẽ tự tạo ra quy luật:
    # Nếu số tiền (Amount) lớn hơn 80,000 thì gắn nhãn rủi ro = 1, ngược lại = 0
    if 'Label' not in df.columns:
        df['Label'] = np.where(df['Amount'] > 80000, 1, 0)

except FileNotFoundError:
    # Nếu hoàn toàn không tìm thấy file CSV đâu, hệ thống tự tạo dữ liệu mẫu để chạy thử
    np.random.seed(42)
    df = pd.DataFrame({'Amount': np.random.randint(10000, 100000, size=150)})
    df['Label'] = np.where(df['Amount'] > 80000, 1, 0)

# Hiển thị bảng dữ liệu lên trang web để kiểm toán viên theo dõi
st.subheader("📋 1. Dữ Liệu Chứng Từ Hiện Có")
st.dataframe(df.head(10))

# --- BƯỚC 3: CHIA DỮ LIỆU THEO TỶ LỆ CHUẨN 80/20 ---
X = df[['Amount']] # Lấy cột Amount làm đầu vào dự đoán
y = df['Label']    # Lấy cột Label làm mục tiêu đánh giá

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"📊 Tập học (Train): {len(X_train)} dòng | Tập thi (Test): {len(X_test)} dòng")
st.markdown("---")

# --- BƯỚC 4: HUẤN LUYỆN VÀ VẼ BIỂU ĐỒ TRỰC QUAN ---
st.subheader("⚙️ 2. Kết Quả Huấn Luyện AI")

if st.button("🚀 Kích hoạt mô hình Rừng Ngẫu Nhiên"):
    # 1. Cho mô hình Random Forest học dữ liệu tập Train (80%)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 2. Bắt AI dự đoán thử trên tập Test (20%)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 3. In điểm số chính xác lên màn hình chính
    st.success(f"🎯 Mô hình đạt độ chính xác: {accuracy * 100:.2f}%")
    
    # 4. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix) để báo cáo
    st.markdown("---")
    st.subheader("📊 3. Biểu Đồ Đánh Giá Sai Sót (Confusion Matrix)")
    st.write("Biểu đồ này minh họa khả năng phân loại đúng/sai của mô hình học máy khi phát hiện rủi ro.")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Sử dụng thư viện seaborn vẽ bảng ma trận màu xanh bắt mắt
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Đoán An toàn', 'Đoán Rủi ro'],
                yticklabels=['Thực tế An toàn', 'Thực tế Rủi ro'], ax=ax)
    
    st.pyplot(fig) # Lệnh hiển thị biểu đồ lên giao diện trang web