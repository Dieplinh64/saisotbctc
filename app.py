import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Kiểm Toán AI Cao Cấp", layout="wide")
st.title("🛡️ HỆ THỐNG KIỂM TOÁN AI ĐA CHIỀU (RANDOM FOREST)")

# Cổng tải file trực tiếp trên giao diện web
uploaded_file = st.file_uploader("Kéo thả file Excel dữ liệu kiểm toán vào đây:", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Đọc dữ liệu từ file người dùng kéo vào
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    
    # --- PHẦN 3: LOGIC KIỂM TOÁN (FEATURE ENGINEERING) ---
    # 1. Tính toán sai lệch số dư (Trước - Số tiền = Sau)
    if all(col in df.columns for col in ['Balance_Before', 'Amount', 'Balance_After']):
        df['Discrepancy'] = (df['Balance_Before'] - df['Amount']) - df['Balance_After']
    else:
        df['Discrepancy'] = 0

    # 2. Xử lý thời gian (Giao dịch ban đêm từ 23h - 4h)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Is_Night_Transaction'] = np.where((df['Timestamp'].dt.hour >= 23) | (df['Timestamp'].dt.hour <= 4), 1, 0)
    else:
        df['Is_Night_Transaction'] = 0

    # 3. Gán nhãn rủi ro ranh giới (Kết hợp sai lệch số dư HOẶC giao dịch đêm số tiền lớn hơn phân vị 95%)
    df['Is_Anomaly'] = np.where((df['Discrepancy'].abs() > 0.01) | 
                                ((df['Is_Night_Transaction'] == 1) & (df['Amount'] > df['Amount'].quantile(0.95))), 1, 0)

    # --- PHẦN 4: HUẤN LUYỆN MÔ HÌNH AI (MACHINE LEARNING LAYER) ---
    # Chọn các cột đặc trưng hành vi làm đầu vào cho AI
    features = ['Amount', 'Balance_Before', 'Balance_After', 'Is_Night_Transaction']
    X = df[features].fillna(0)  # Ma trận đề bài (Làm sạch các ô trống bằng số 0)
    y = df['Is_Anomaly']       # Vectơ đáp án mẫu (0: An toàn, 1: Cờ đỏ)
    
    # Sử dụng hàm train_test_split để chia tách tập Học (70%) và tập Thi (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Khởi tạo mô hình Rừng ngẫu nhiên với 100 cây quyết định toán học độc lập
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)  # Kích hoạt lệnh ép AI phải học thuộc quy luật dữ liệu

    # --- PHẦN 5: GIAO DIỆN HIỂN THỊ KẾT QUẢ VÀ TRỰC QUAN HÓA ---
    st.subheader("📊 Kết quả phân tích từ mô hình AI")
    # Tính toán chỉ số chính xác dựa trên bài thi thử của AI
    st.write(f"Độ chính xác mô hình: **{rf.score(X_test, y_test)*100:.2f}%**")
    
    # Vẽ biểu đồ chấm phân tán phân tách màu sắc
    fig = px.scatter(
        df, x="Balance_Before", y="Amount", color=df['Is_Anomaly'].astype(str),
        title="Biểu đồ phân loại: Chấm ĐỎ là giao dịch AI phát hiện bất thường",
        color_discrete_map={"0": "green", "1": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Xuất ra bảng danh sách đen chứa các dòng lỗi để kiểm toán viên hậu kiểm
    st.subheader("📋 Danh sách chi tiết các giao dịch cần hậu kiểm (Cờ đỏ)")
    st.dataframe(df[df['Is_Anomaly'] == 1])
