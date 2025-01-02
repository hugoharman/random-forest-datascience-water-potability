import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib  
import os
from PIL import Image

MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'standard_scaler.pkl'

def load_data():
    df = pd.read_csv('water_potability.csv')
    return df.fillna(df.mean())

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

def load_model_and_scaler():
    """Load the model and scaler from disk if they exist."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        return None, None

def main():
    st.title('Analisis Kelayakan Air Minum - A11.2021.13937')
    
    # Load data
    df = load_data()
    
    st.header('Informasi Dataset')
    st.write(df.head())
    st.write('Ukuran Dataset:', df.shape)
    
    # Visualisasi Data
    st.header('Visualisasi Data')
    
    st.subheader('Matriks Korelasi')
    corr_image = load_image('correlation_heatmap.png')
    st.image(corr_image, caption='Matriks Korelasi')
    
    st.subheader('Distribusi Fitur')
    dist_image = load_image('feature_distributions.png')
    st.image(dist_image, caption='Distribusi Fitur')
    
    model, scaler = load_model_and_scaler()
    
    if model is not None and scaler is not None:
        st.header('Model Performance')
        
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        
        X_scaled = scaler.transform(X)
        
        y_pred = model.predict(X_scaled)
        
        st.subheader('Performa Model')
        st.write('Laporan Klasifikasi:')
        st.code(classification_report(y, y_pred))
        
        st.subheader('Tingkat Kepentingan Fitur')
        importance_df = pd.DataFrame({
            'Fitur': X.columns,
            'Tingkat_Kepentingan': model.feature_importances_
        }).sort_values('Tingkat_Kepentingan', ascending=False)
        
        st.bar_chart(importance_df.set_index('Fitur'))
        
        # Confusion matrix
        st.subheader('Matriks Konfusi')
        cm_image = load_image('confusion_matrix.png')
        st.image(cm_image, caption='Matriks Konfusi')
    else:
        st.error('Model belum dilatih')
    
    st.header('Buat Prediksi')
    st.write('Masukkan nilai untuk memprediksi kelayakan air:')
    
    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f'Masukkan nilai {column}', value=float(df[column].mean()))
    
    if st.button('Prediksi'):
        if model is not None and scaler is not None:
            # Ubah dan skala data input
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            st.subheader('Hasil Prediksi')
            st.write('Air:', 'Layak Minum' if prediction[0] == 1 else 'Tidak Layak Minum')
            st.write('Tingkat Keyakinan:', f'{max(probability[0]) * 100:.2f}%')
        else:
            st.error('Model belum dilatih. Silakan latih model terlebih dahulu.')

if __name__ == '__main__':
    main()