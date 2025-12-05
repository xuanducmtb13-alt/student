
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# --- Load model, scaler, feature columns ---
with gzip.open("random_forest_model.pkl.gz", "rb") as f:
    rf_model = joblib.load(f)
with gzip.open("scaler.pkl.gz", "rb") as f:
    scaler = joblib.load(f)
with gzip.open("feature_columns.pkl.gz", "rb") as f:
    feature_columns = joblib.load(f)

# --- Nhập dữ liệu người dùng ---
StudyHours = st.number_input("Số giờ học mỗi tuần", 0, 100, 10)
Attendance = st.slider("Tỷ lệ chuyên cần (%)", 0, 100, 80)
Resources = st.slider("Sử dụng tài nguyên học tập (%)", 0, 100, 50)
Motivation = st.slider("Mức độ động lực (0-10)", 0, 10, 7)
Age = st.number_input("Tuổi", 5, 30, 18)
OnlineCourses = st.number_input("Số khóa học trực tuyến tham gia", 0, 50, 5)
AssignmentCompletion = st.slider("Hoàn thành bài tập (%)", 0, 100, 80)

Extracurricular = st.selectbox("Hoạt động ngoại khóa", ["Không", "Có"])
Extracurricular = 1 if Extracurricular == "Có" else 0
Internet = st.selectbox("Có Internet không?", ["Không", "Có"])
Internet = 1 if Internet == "Có" else 0
Gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
Gender = 1 if Gender == "Nam" else 0
Discussions = st.selectbox("Tham gia thảo luận", ["Không", "Có"])
Discussions = 1 if Discussions == "Có" else 0
EduTech = st.selectbox("Sử dụng EduTech", ["Không", "Có"])
EduTech = 1 if EduTech == "Có" else 0

LearningStyle = st.selectbox("Phong cách học tập", ["Visual", "Auditory", "Kinesthetic"])
StressLevel = st.selectbox("Mức độ căng thẳng", ["Low", "Medium", "High"])
FinalGrade = st.selectbox("Điểm cuối kỳ", ["A", "B", "C", "D", "F"])

# --- Tạo dataframe input raw ---
input_raw = pd.DataFrame({
    'StudyHours':[StudyHours],
    'Attendance':[Attendance],
    'Resources':[Resources],
    'Motivation':[Motivation],
    'Age':[Age],
    'OnlineCourses':[OnlineCourses],
    'AssignmentCompletion':[AssignmentCompletion],
    'Extracurricular':[Extracurricular],
    'Internet':[Internet],
    'Gender':[Gender],
    'Discussions':[Discussions],
    'EduTech':[EduTech],
    'LearningStyle':[LearningStyle],
    'StressLevel':[StressLevel],
    'FinalGrade':[FinalGrade]
})

# --- One-hot giống khi train ---
input_processed = pd.get_dummies(input_raw, columns=["LearningStyle", "StressLevel", "FinalGrade"], drop_first=True)

# --- Scale numeric ---
numeric_cols = ["StudyHours", "Attendance", "Resources", "Motivation", "Age", "OnlineCourses", "AssignmentCompletion"]
input_processed[numeric_cols] = scaler.transform(input_processed[numeric_cols])

# --- Thêm các cột thiếu, đảm bảo đúng feature_columns ---
for col in feature_columns:
    if col not in input_processed.columns:
        input_processed[col] = 0
input_processed = input_processed[feature_columns]  # Sắp xếp đúng thứ tự

# --- Predict ---
if st.button("Dự đoán Điểm Thi"):
    prediction = rf_model.predict(input_processed)
    st.success(f"Điểm dự đoán: {prediction[0]:.2f}")

