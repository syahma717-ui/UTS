import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# 🎯 CONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="AI Vision Dashboard", page_icon="🧠", layout="wide")

# ==========================
# 🌈 CUSTOM CSS FUTURISTIK
# ==========================
st.markdown("""
    <style>
    /* Background dan Font */
    body {
        background-color: #0f1116;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Judul utama */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00eaff;
        text-shadow: 0 0 20px #00eaff;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #b0b0b0;
        margin-bottom: 30px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0f14 0%, #12161e 100%);
        border-right: 1px solid #00eaff50;
    }

    /* Tombol upload */
    div[data-testid="stFileUploader"] {
        background: #161a24;
        border-radius: 10px;
        border: 2px dashed #00eaff80;
        padding: 15px;
    }

    /* Gambar hasil */
    img {
        border-radius: 10px;
        box-shadow: 0 0 20px #00eaff30;
    }

    /* Spinner */
    .stSpinner > div {
        color: #00eaff !important;
    }

    /* Card-style box */
    .card {
        background: #161a24;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 25px #00eaff20;
        margin-bottom: 25px;
    }

    /* Diagram */
    canvas {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🚀 LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/Syahma_Laporan_4.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🧠 HEADER
# ==========================
st.markdown('<h1 class="main-title">🧠 AI Vision Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi Objek & Klasifikasi Gambar — Futuristic Edition</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================
# 🎛️ SIDEBAR
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🧬 Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar untuk memulai analisis")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg","jpeg","png"])

# ==========================
# ⚙️ PROSES GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Gambar yang diunggah", use_container_width=True)
    st.markdown("---")

    # ==========================
    # 📦 YOLO DETEKSI
    # ==========================
    if menu == "📦 Deteksi Objek (YOLO)":
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)

    # ==========================
    # 🧬 KLASIFIKASI
    # ==========================
    elif menu == "🧬 Klasifikasi Gambar":
        with st.spinner("🧠 Sedang mengklasifikasikan gambar..."):
            try:
                H, W, C = classifier.input_shape[1:4]
                img = img.convert('RGB') if C == 3 else img.convert('L')
                img_resized = img.resize((W, H))
                img_array = np.array(img_resized).astype('float32') / 255.0
                if C == 1 and img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = classifier.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                st.markdown(f'<div class="card"><h3>🎯 Hasil Prediksi</h3>'
                            f'<p><b>Kelas:</b> {predicted_class}</p>'
                            f'<p><b>Tingkat Kepercayaan:</b> {confidence*100:.2f}%</p></div>',
                            unsafe_allow_html=True)

                # Chart probabilitas
                class_probs = prediction[0]
                classes = [f"Kelas {i}" for i in range(len(class_probs))]
                fig, ax = plt.subplots()
                ax.bar(classes, class_probs*100, color='#00eaff')
                ax.set_ylabel("Probabilitas (%)", color='white')
                ax.set_title("Distribusi Probabilitas", color='white')
                ax.tick_params(colors='white')
                fig.patch.set_facecolor('#161a24')
                st.pyplot(fig)

                # Statistik Model
                total_params = classifier.count_params()
                trainable_params = np.sum([tf.keras.backend.count_params(w) for w in classifier.trainable_weights])
                non_trainable_params = total_params - trainable_params

                st.markdown(f"""
                <div class="card">
                    <h3>📊 Statistik Model</h3>
                    <p>Jumlah kelas: {classifier.output_shape[1]}</p>
                    <p>Total parameter: {total_params:,}</p>
                    <p>Trainable parameter: {trainable_params:,}</p>
                    <p>Non-trainable parameter: {non_trainable_params:,}</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
else:
    st.info("💡 Silakan unggah gambar terlebih dahulu untuk memulai prediksi.")

# ==========================
# ⚡ FOOTER
# ==========================
st.markdown('<p class="footer">© 2025 | Dibuat oleh <b>Syahma</b> — Ujian Tengah Semester BIG DATA</p>', unsafe_allow_html=True)
