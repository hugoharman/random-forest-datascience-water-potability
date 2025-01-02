import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib  
import os

MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'standard_scaler.pkl'

def load_data():
    df = pd.read_csv('water_potability.csv')
    return df.fillna(df.mean())

def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriks Korelasi')
    return fig

def plot_feature_distributions(df):
    features = df.drop('Potability', axis=1).columns
    fig = plt.figure(figsize=(15, 10))
    for i, col in enumerate(features, 1):
        plt.subplot(3, 3, i)
        plt.hist(df[col], bins=30)
        plt.title(f'Distribusi {col}')
    plt.tight_layout()
    return fig

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
    corr_fig = plot_correlation(df)
    st.pyplot(corr_fig)
    
    st.subheader('Distribusi Fitur')
    dist_fig = plot_feature_distributions(df)
    st.pyplot(dist_fig)
    
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
        cm_fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        st.pyplot(cm_fig)
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