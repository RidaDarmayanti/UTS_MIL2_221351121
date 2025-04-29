import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl') 

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="stroke_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Risiko Stroke")
st.write("Masukkan data diri Anda untuk memprediksi risiko stroke.")

# Form input pengguna
gender = st.selectbox("Gender", ["Laki-laki", "Perempuan", "Lainnya"])
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Memiliki hipertensi?", ["Tidak", "Ya"])
heart_disease = st.selectbox("Memiliki penyakit jantung?", ["Tidak", "Ya"])
ever_married = st.selectbox("Pernah menikah?", ["Tidak", "Ya"])
work_type = st.selectbox("Jenis pekerjaan", ["Anak-anak", "Pekerjaan Pemerintah", "Swasta", "Wiraswasta", "Tidak Bekerja"])
residence_type = st.selectbox("Tinggal di", ["Perkotaan", "Pedesaan"])
avg_glucose_level = st.slider("Rata-rata kadar glukosa", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.slider("BMI", min_value=0.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox("Status merokok", ["Tidak pernah merokok", "Sebelumnya merokok", "Merokok", "Tidak diketahui"])

# Mapping kategori ke angka
gender_map = {"Laki-laki": 1, "Perempuan": 0, "Lainnya": 2}
hypertension_map = {"Tidak": 0, "Ya": 1}
heart_disease_map = {"Tidak": 0, "Ya": 1}
ever_married_map = {"Tidak": 0, "Ya": 1}
work_type_map = {"Anak-anak": 0, "Pekerjaan Pemerintah": 1, "Swasta": 2, "Wiraswasta": 3, "Tidak Bekerja": 4}
residence_type_map = {"Perkotaan": 1, "Pedesaan": 0}
smoking_status_map = {"Tidak pernah merokok": 1, "Sebelumnya merokok": 2, "Merokok": 3, "Tidak diketahui": 0}

if st.button("Prediksi Risiko Stroke"):
    # Preprocessing input
    input_data = np.array([[ 
        gender_map[gender], 
        age, 
        hypertension_map[hypertension],
        heart_disease_map[heart_disease],
        ever_married_map[ever_married],
        work_type_map[work_type],
        residence_type_map[residence_type],
        avg_glucose_level,
        bmi,
        smoking_status_map[smoking_status]
    ]])

    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = (prediction > 0.7).astype(int)[0][0]  # Binary classification

    # Jika pakai label encoder untuk output (opsional)
    # result_label = label_encoder.inverse_transform([predicted_class])[0]

    if predicted_class == 1:
        st.error("⚠️ Hasil Prediksi: Berisiko Stroke!")
    else:
        st.success("✅ Hasil Prediksi: Tidak Berisiko Stroke.")
