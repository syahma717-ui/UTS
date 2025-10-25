import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2


# ==========================
# Load Models
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
st.set_page_config(
    page_title="Dashboard Deteksi & Klasifikasi Gambar",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dashboard Deteksi & Klasifikasi Gambar")
st.markdown("*Dibuat oleh Syahma — Laporan 4 BIG DATA*")
st.markdown("---")

# ==========================
# 🧭 SIDEBAR MENU
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🧬 Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar di bawah untuk melakukan prediksi")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🔍 PROSES & OUTPUT
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    if menu == "📦 Deteksi Objek (YOLO)":
        if yolo_model is not None:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
                st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)
        else:
            st.warning("Model YOLO belum berhasil dimuat!")

    elif menu == "🧬 Klasifikasi Gambar":
        if classifier is not None:
            with st.spinner("🧠 Sedang mengklasifikasikan gambar..."):
                # Preprocessing
                img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success("✅ Klasifikasi Berhasil!")
                st.write("### Hasil Prediksi:", class_index)
                st.write("Probabilitas:", f"{confidence:.2f}")
        else:
            st.warning("Model klasifikasi belum berhasil dimuat!")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai.")
# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.caption("© 2025 | Dashboard dibuat untuk Ujian Tengah Semester BIG DATA oleh Syahma")
