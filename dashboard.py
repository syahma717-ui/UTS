# ==========================
# 📦 IMPORT LIBRARY
# ==========================
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# ⚙️ LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model/Syahma_Laporan_4.h5")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        return None

classifier = load_model()

# ==========================
# 🖥️ STREAMLIT UI
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi Gambar",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dashboard Klasifikasi Gambar")
st.markdown("**Dibuat oleh Syahma — Laporan 4 BIG DATA**")
st.markdown("---")

# ==========================
# 🧭 SIDEBAR MENU
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["🧬 Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar untuk melakukan klasifikasi menggunakan model TensorFlow")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🔍 PROSES & OUTPUT
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    if classifier is not None:
        with st.spinner("🧠 Sedang mengklasifikasikan gambar..."):
            # Preprocessing gambar
            img_resized = img.resize((224, 224))  # sesuaikan dengan ukuran input model
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Tampilkan hasil
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
st.caption("© 2025 | Dashboard klasifikasi gambar oleh Syahma")
