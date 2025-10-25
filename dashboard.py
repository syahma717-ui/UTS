# bagian paling atas file dashboard.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# coba import ultralytics & cv2, tapi jangan crash bila gagal
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    yolo_load_error = str(e)

try:
    import cv2
except Exception:
    # cv2 mungkin tidak tersedia (tapi kita sudah menambahkan opencv-python-headless)
    pass

# load classifier (TensorFlow) seperti biasa dengan try/except
@st.cache_resource
def load_models():
    classifier = None
    try:
        classifier = tf.keras.models.load_model("model/Syahma_Laporan_4.h5")
    except Exception as e:
        st.error(f"Gagal memuat classifier: {e}")
    yolo_model = None
    if ULTRALYTICS_AVAILABLE:
        try:
            yolo_model = YOLO("model/best.pt")
        except Exception as e:
            # jangan crash, kita hanya menampilkan warning
            yolo_model = None
            yolo_load_error = str(e)
    return yolo_model, classifier

yolo_model, classifier = load_models()
menu_items = ["🧬 Klasifikasi Gambar"]
if ULTRALYTICS_AVAILABLE:
    menu_items.insert(0, "📦 Deteksi Objek (YOLO)")
menu = st.sidebar.selectbox("Pilih Mode:", menu_items)
