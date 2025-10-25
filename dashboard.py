import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# 🌸 Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="Aesthetic AI Dashboard", page_icon="💫", layout="wide")

# ==========================
# 🎨 Custom CSS (Glassmorphism + Pastel Style)
# ==========================
st.markdown("""
    <style>
    /* Background soft gradient */
    body {
        background: linear-gradient(135deg, #f7f9ff 0%, #e7f0ff 50%, #f8f5ff 100%);
        font-family: 'Poppins', sans-serif;
        color: #333;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #6a9cfb, #b07bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid #ddd;
    }

    /* Upload box */
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.6);
        border-radius: 15px;
        border: 2px dashed #b5c7ff;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #8aa8ff;
        background: rgba(255,255,255,0.8);
    }

    /* Card style */
    .card {
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.6);
    }

    /* Chart */
    canvas {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 13px;
        color: #777;
        margin-top: 40px;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(90deg, #82aaff, #b892ff) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }

    img {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🧠 Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/Syahma_Laporan_4.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🌼 Header
# ==========================
st.markdown('<h1 class="main-title">💫 Aesthetic AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi Objek & Klasifikasi Gambar — Clean & Soft Style</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================
# 🎛️ Sidebar
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🧬 Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar untuk memulai analisis")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg","jpeg","png"])

# ==========================
# ⚙️ Proses Gambar
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)
    st.markdown("---")

    # ==========================
    # 📦 Mode YOLO
    # ==========================
    if menu == "📦 Deteksi Objek (YOLO)":
        with st.spinner("✨ Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)

    # ==========================
    # 🧬 Mode Klasifikasi
    # ==========================
    elif menu == "🧬 Klasifikasi Gambar":
        with st.spinner("🌸 Sedang mengklasifikasikan gambar..."):
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

                # Card hasil prediksi
                st.markdown(f"""
                <div class="card">
                    <h3>🌼 Hasil Klasifikasi</h3>
                    <p><b>Kelas:</b> {predicted_class}</p>
                    <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Probabilitas kelas
                class_probs = prediction[0]
                classes = [f"Kelas {i}" for i in range(len(class_probs))]
                fig, ax = plt.subplots()
                ax.bar(classes, class_probs*100, color='#91a8ff')
                ax.set_ylabel("Probabilitas (%)")
                ax.set_title("Distribusi Probabilitas")
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
    st.info("🌷 Silakan unggah gambar untuk memulai prediksi.")

# ==========================
# 🌸 Footer
# ==========================
st.markdown('<p class="footer">© 2025 | Dibuat oleh <b>Syahma</b> — Ujian Tengah Semester BIG DATA</p>', unsafe_allow_html=True)
