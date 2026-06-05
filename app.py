import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Kiểm Toán AI Cao Cấp", layout="wide")
st.title("🛡️ HỆ THỐNG KIỂM TOÁN AI (KIỂM TRA SỐ DƯ & GIAO DỊCH BAN ĐÊM)")

# 2. TẢI FILE
uploaded_file = st.file_uploader("Kéo thả file Excel dữ liệu kiểm toán vào đây:", type=["xlsx", "csv"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    
    # --- PHẦN TÍNH TOÁN LOGIC KIỂM TOÁN ---
    # 1. Kiểm tra sai lệch số dư
    if all(col in df.columns for col in ['Balance_Before', 'Amount', 'Balance_After']):
        df['Discrepancy'] = (df['Balance_Before'] - df['Amount']) - df['Balance_After']
    else:
        df['Discrepancy'] = 0

    # 2. Tạo cột Giao dịch ban đêm (Giả định cột 'Timestamp' có dạng giờ)
    # Nếu dữ liệu có giờ, AI sẽ tự phân loại: 23h - 4h là ban đêm (1)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['Is_Night_Transaction'] = np.where((df['Hour'] >= 23) | (df['Hour'] <= 4), 1, 0)
    
    # 3. GÁN NHÃN GIAN LẬN (Để AI học)
    # Lỗi nếu: Sai số dư HOẶC Giao dịch đêm + Số tiền quá lớn
    df['Is_Anomaly'] = np.where((df['Discrepancy'].abs() > 0.01) | 
                                ((df['Is_Night_Transaction'] == 1) & (df['Amount'] > df['Amount'].quantile(0.95))), 1, 0)

    # --- PHẦN AI HỌC MÔ HÌNH ---
    features = ['Amount', 'Balance_Before', 'Balance_After', 'Is_Night_Transaction']
    X = df[features].fillna(0)
    y = df['Is_Anomaly']
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    
    # 4. HIỂN THỊ KẾT QUẢ
    st.subheader("📊 Báo cáo phân tích AI")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng giao dịch bị AI gắn cờ đỏ", f"{df['Is_Anomaly'].sum()} ca")
    with col2:
        st.metric("Giao dịch ban đêm nghi vấn", f"{df['Is_Night_Transaction'].sum()} ca")

    # 5. BIỂU ĐỒ TRỰC QUAN
    fig = px.scatter(
        df, x="Is_Night_Transaction", y="Amount", color=df['Is_Anomaly'].astype(str),
        title="AI phân loại: Giao dịch ban đêm (1) vs Ban ngày (0) - Chấm ĐỎ là nghi vấn",
        color_discrete_map={"0": "green", "1": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6. BẢNG CHI TIẾT
    st.subheader("📋 Danh sách giao dịch AI cảnh báo")
    st.dataframe(df[df['Is_Anomaly'] == 1])