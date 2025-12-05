
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip 

# Load mô hình Random Forest đã huấn luyện
with gzip.open("random_forest_model.pkl.gz", "rb") as f:
    rf_model = joblib.load(f)


# Load mô hình Random Forest đã huấn luyện
with gzip.open("random_forest_model.pkl.gz", "rb") as f:
    rf_model = joblib.load(f)

st.title("Dự đoán Điểm Thi Học Sinh 🎓")
st.write("Ứng dụng dự đoán điểm thi dựa trên các chỉ số học tập, hoạt động ngoại khóa, tâm lý và công nghệ giáo dục.")

st.header("Nhập thông tin học sinh")

# --- Numeric Inputs ---
StudyHours = st.number_input("Số giờ học mỗi tuần", 0, 100, 10)
Attendance = st.slider("Tỷ lệ chuyên cần (%)", 0, 100, 80)
Resources = st.slider("Sử dụng tài nguyên học tập (%)", 0, 100, 50)
Motivation = st.slider("Mức độ động lực (0-10)", 0, 10, 7)
Age = st.number_input("Tuổi", 5, 30, 18)
OnlineCourses = st.number_input("Số khóa học trực tuyến tham gia", 0, 50, 5)
AssignmentCompletion = st.slider("Hoàn thành bài tập (%)", 0, 100, 80)

# --- Binary Inputs ---
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

# --- Categorical Inputs ---
LearningStyle = st.selectbox("Phong cách học tập", ["Visual", "Auditory", "Kinesthetic"])
StressLevel = st.selectbox("Mức độ căng thẳng", ["Low", "Medium", "High"])
FinalGrade = st.selectbox("Điểm cuối kỳ", ["A", "B", "C", "D", "F"])

# --- Chuẩn bị DataFrame đầu vào ---
input_data = pd.DataFrame({
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
})

# --- One-hot encoding cho categorical columns ---
learning_cols = ['LearningStyle_Auditory', 'LearningStyle_Kinesthetic']
stress_cols = ['StressLevel_Medium', 'StressLevel_High']
grade_cols = ['FinalGrade_B', 'FinalGrade_C', 'FinalGrade_D', 'FinalGrade_F']

for col in learning_cols:
    input_data[col] = 1 if LearningStyle in col else 0
for col in stress_cols:
    input_data[col] = 1 if StressLevel in col else 0
for col in grade_cols:
    input_data[col] = 1 if FinalGrade in col else 0

# --- Dự đoán ---
if st.button("Dự đoán Điểm Thi"):
    prediction = rf_model.predict(input_data)
    st.success(f"Điểm dự đoán: {prediction[0]:.2f}")
