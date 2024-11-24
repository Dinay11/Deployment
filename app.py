import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# Load model dan objek pendukung
@st.cache_resource
def load_model():
    with open('svm_model_berhasil.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer_smote.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open('label_encoder.pkl', 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Aplikasi Streamlit
st.title("Prediksi Sentimen Analisis")

st.markdown("""
Upload text data atau masukkan teks langsung.
""")

# Input teks
user_input = st.text_area("Masukkan teks:", placeholder="Contoh: Saya setuju dengan program makan siang gratis.")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Harap masukkan teks untuk prediksi.")
    else:
        # Transformasi input dengan TF-IDF Vectorizer
        input_vectorized = vectorizer.transform([user_input]).toarray()
        
        # Prediksi dengan model
        prediction = model.predict(input_vectorized)
        
        # Decode label jika label encoder digunakan
        predicted_label = label_encoder.inverse_transform(prediction)
        
        st.success(f"Prediksi Kategori: **{predicted_label[0]}**")
