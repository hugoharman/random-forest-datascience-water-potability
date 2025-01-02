import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'standard_scaler.pkl'

def load_data():
    df = pd.read_csv('water_potability.csv')
    df = df.fillna(df.mean())  
    return df

def load_image(image_path):
    return Image.open(image_path)

def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        return None, None

def main():
    st.title('Analisis Kelayakan Air Minum - A11.2021.13937')
    
    df = load_data()
    
    st.header('Informasi Dataset')
    st.write(df.head())
    st.write('Ukuran Dataset:', df.shape)
    
    st.header('Visualisasi Data')
    
    st.subheader('Matriks Korelasi')
    corr_image = load_image('correlation_heatmap.png')
    st.image(corr_image, caption='Matriks Korelasi')
    
    st.subheader('Distribusi Fitur')
    dist_image = load_image('feature_distributions.png')
    st.image(dist_image, caption='Distribusi Fitur')
    
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error('Model belum dilatih')
        return
        
    X = df.drop('Potability', axis=1)
    
    st.header('Buat Prediksi')
    st.write('Masukkan nilai untuk memprediksi kelayakan air:')
    
    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f'Masukkan nilai {column}', value=float(df[column].mean()))
    
    if st.button('Prediksi'):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        st.subheader('Hasil Prediksi')
        st.write('Air:', 'Layak Minum' if prediction[0] == 1 else 'Tidak Layak Minum')
        st.write('Tingkat Keyakinan:', f'{max(probability[0]) * 100:.2f}%')

if __name__ == '__main__':
    main()