import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models Deteksi
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Syahma_Laporan_4.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🖥 STREAMLIT UI 
# ==========================
st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🧠", layout="centered")
st.title("🧠 Dashboard deteksi & Klasifikasi Gambar")
uploaded_file = st.file_uploader("📤 Unggah gambar:", type=["jpg","jpeg","png"])


st.markdown("*Dibuat oleh Syahma — Ujuan Tengah Semester BIG DATA*")
st.markdown("---")
# ==========================
# 🧭 SIDEBAR MENU
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🧬 Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar di bawah untuk melakukan prediksi")

# Satu file uploader global
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    if menu == "📦 Deteksi Objek (YOLO)":
        if yolo_model is not None:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)
        else:
            st.warning("Model YOLO belum berhasil dimuat!")

  uploaded_file = st.file_uploader("📤 Unggah gambar:", type=["jpg","jpeg","png"])

if uploaded_file is not None and classifier is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    try:
        H, W, C = classifier.input_shape[1:4]
        if C == 3:
            img = img.convert('RGB')
        elif C == 1:
            img = img.convert('L')

        img_resized = img.resize((W, H))
        img_array = np.array(img_resized).astype('float32') / 255.0
        if C == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = classifier.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Hasil Prediksi: Kelas {predicted_class} ({confidence*100:.2f}%)")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
else:
    st.info("Silakan unggah gambar untuk memulai klasifikasi.")

  
# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.caption("© 2025 | Dashboard dibuat untuk Ujian Tengah Semester BIG DATA oleh Syahma")

