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
# 🖥 STREAMLIT UI Deteksi
# ==========================
st.set_page_config(
    page_title="Dashboard Deteksi & Klasifikasi Gambar",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dashboard Deteksi & Klasifikasi Gambar")
st.markdown("*Dibuat oleh Syahma — Ujuan Tengah Semester BIG DATA*")
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
        if uploaded_file is not None:
            with st.spinner("🧠 Sedang mengklasifikasikan gambar..."):
                try:
                    # ==========================
                    # Load gambar
                    # ==========================
                    img = Image.open(uploaded_file)

                    # ==========================
                    # Ambil ukuran input model
                    # ==========================
                    target_height = classifier.input_shape[1]
                    target_width = classifier.input_shape[2]
                    target_channels = classifier.input_shape[3]

                    # ==========================
                    # Sesuaikan channel
                    # ==========================
                    if target_channels == 3:
                        img = img.convert('RGB')
                    elif target_channels == 1:
                        img = img.convert('L')

                    # ==========================
                    # Resize dan preprocessing
                    # ==========================
                    img_resized = img.resize((target_width, target_height))
                    img_array = np.array(img_resized).astype('float32') / 255.0

                    # Jika grayscale, tambahkan channel axis
                    if target_channels == 1 and img_array.ndim == 2:
                        img_array = np.expand_dims(img_array, axis=-1)

                    # Tambahkan batch dimension
                    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1,H,W,C)

                    # ==========================
                    # Prediksi
                    # ==========================
                    prediction = classifier.predict(img_array)  # shape (1, num_classes)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # ==========================
                    # Tampilkan gambar dan hasil prediksi
                    # ==========================
                    st.image(img, caption="Gambar yang diunggah", use_column_width=True)
                    st.success(f"Hasil Prediksi: **Kelas {predicted_class}**")
                    st.write(f"Tingkat Kepercayaan Kelas Terpilih: **{confidence:.2f}**")

                    # ==========================
                    # Tampilkan probabilitas semua kelas
                    # ==========================
                    st.write("Probabilitas untuk semua kelas:")
                    num_classes = prediction.shape[1]
                    for i in range(num_classes):
                        st.write(f"Kelas {i}: {prediction[0, i]:.4f}")

                    # ==========================
                    # Statistik model
                    # ==========================
                    total_params = classifier.count_params()
                    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in classifier.trainable_weights])
                    non_trainable_params = total_params - trainable_params

                    st.write("📊 Statistik Model:")
                    st.write(f"Total parameter: {total_params}")
                    st.write(f"Trainable parameter: {trainable_params}")
                    st.write(f"Non-trainable parameter: {non_trainable_params}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.info("Silakan unggah gambar untuk memulai klasifikasi.")
    else:
        st.error("Model classifier belum berhasil dimuat.")
  
# ==========================
# 📚 FOOTER
# ==========================
st.markdown("---")
st.caption("© 2025 | Dashboard dibuat untuk Ujian Tengah Semester BIG DATA oleh Syahma")

